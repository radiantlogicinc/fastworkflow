import os
import time


def add_two_numbers(a: float, b: float) -> float:
    if call_log := os.environ.get("FW_TEST_ADD_CALL_LOG"):
        with open(call_log, "a") as fh:
            fh.write("call\n")
    sleep_seconds = float(os.environ.get("FW_TEST_ADD_SLEEP_SECONDS", "0") or "0")
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return a + b
