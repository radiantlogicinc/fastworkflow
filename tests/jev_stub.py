"""A loopback stand-in for TypeSafe's Jev API, spoken to over real HTTP by the real ``typesafe_sdk``.

It serves ``POST /v1/systemone``. Every question is answered by ``answer(name,
question)`` -- a float is a noul, a string a choice, a dict the raw answer --
unless ``respond(body)`` returns ``(status, payload)`` or ``(status, payload,
headers)`` for that request, which is how error envelopes are scripted.
``delay`` holds every response back that many seconds; ``stall`` sends no
byte at all until the stub is closed or the client hangs up; ``trickle =
(slices, interval)`` sends the headers at once and then the body in that many
slices, *interval* seconds apart -- each slice resets a per-read timeout, so
only a wall-clock bound ends it early (or ``end_trickles``, which cuts every
body being trickled and closes its connection). Each request's path,
headers and JSON body are recorded in ``requests`` before the reply, and each
reply carries its own ``x-typesafe-request-id`` (``req-1``, ``req-2``, ...).

A client that gives up waiting (a timeout) and disconnects is not an error
here: the reply is dropped silently. ``close`` wakes a delayed reply at once.
Bound to 127.0.0.1 on a free port; features reach it through
``FW_JEV_BASE_URL`` (the ``jev_stub`` fixture in ``tests/conftest.py``).
"""
from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

SYSTEM_ONE_PATH = "/v1/systemone"
REQUEST_ID_HEADER = "x-typesafe-request-id"
_DISCONNECTS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout)


def default_answer(name: str, question: dict[str, Any]) -> Any:
    """Yes (0.9) to every noul; the first criterion of every choice."""
    if question.get("type") == "choice":
        return next(iter(question["criteria"]))
    return 0.9


def choice_answer(choice: str, criteria: Any, p_choice: float = 0.97) -> dict[str, Any]:
    """A choice answer picking *choice*, the rest of the probability spread over the other criteria."""
    others = [key for key in criteria if key != choice]
    rest = (1.0 - p_choice) / len(others) if others else 0.0
    return {"type": "choice", "choice": choice, "confidence": p_choice,
            "probabilities": {key: (p_choice if key == choice else rest) for key in criteria}}


def _as_answer(value: Any, question: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return choice_answer(value, question.get("criteria") or {value: ""})
    return {"type": "noul", "noul": float(value)}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that timed out and hung up is expected; anything else is reported as usual.
        if isinstance(sys.exc_info()[1], _DISCONNECTS):
            return
        super().handle_error(request, client_address)


class JevStub:
    """One running stand-in; ``close`` it (the fixture does)."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.answer: Callable[[str, dict[str, Any]], Any] = default_answer
        self.respond: Optional[Callable[[Any], Optional[tuple]]] = None
        self.delay = 0.0
        self.stall = False
        self.trickle: Optional[tuple[int, float]] = None
        self._trickle_epoch = 0
        self._served = 0
        self._lock = threading.Lock()
        self._closing = threading.Event()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
                stub._serve(self)

        self._server = _Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="jev-stub", daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    @property
    def bodies(self) -> list[Any]:
        with self._lock:
            return [request["body"] for request in self.requests]

    def end_trickles(self) -> None:
        """Stop trickling: bodies being trickled stop at their next slice and the connection closes."""
        with self._lock:
            self.trickle = None
            self._trickle_epoch += 1

    def close(self) -> None:
        self._closing.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _serve(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            length = int(handler.headers.get("Content-Length") or 0)
            raw_body = handler.rfile.read(length) if length else b""
            body = json.loads(raw_body) if raw_body else None
            with self._lock:
                self.requests.append({"path": handler.path, "headers": dict(handler.headers), "body": body})
                self._served += 1
                request_id = f"req-{self._served}"
                trickle, epoch = self.trickle, self._trickle_epoch
            if self.stall:
                self._closing.wait()
                return
            if self.delay:
                self._closing.wait(self.delay)
            status, payload, headers = self._reply(handler.path, body)
            raw = json.dumps(payload).encode("utf-8")
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(raw)))
            handler.send_header(REQUEST_ID_HEADER, request_id)
            for name, value in headers.items():
                handler.send_header(name, value)
            handler.end_headers()
            if trickle:
                slices, interval = trickle
                step = max(1, -(-len(raw) // max(1, int(slices))))
                for start in range(0, len(raw), step):
                    if start and (self._closing.wait(interval) or self._trickle_epoch != epoch):
                        break
                    handler.wfile.write(raw[start:start + step])
                    handler.wfile.flush()
                handler.close_connection = True
                return
            handler.wfile.write(raw)
        except _DISCONNECTS:
            handler.close_connection = True

    def _reply(self, path: str, body: Any) -> tuple[int, Any, dict[str, str]]:
        if path != SYSTEM_ONE_PATH:
            return 404, {"error": {"type": "not_found", "message": f"No route {path}."}}, {}
        if self.respond is not None:
            scripted = self.respond(body)
            if scripted is not None:
                status, payload, *rest = scripted
                return status, payload, (rest[0] if rest else {})
        questions = body.get("questions") or {}
        answers = {name: _as_answer(self.answer(name, question), question)
                   for name, question in questions.items()}
        return 200, {"model": body.get("model"), "answers": answers,
                     "usage": {"input_tokens": 7, "output_tokens": 1}}, {}
