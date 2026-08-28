"""fastWorkflow Chatbot — debug-mode viewer for the observability DB.

Design: docs/fastworkflow_observability_studio_design.md §3.4.

The debug-mode HTTP layer is stdlib-only by design [R23]: it must work on a
base install without the ``[server]`` extra. Nothing in this package may
import FastAPI/uvicorn or any other third-party HTTP dependency. The SPA is
one self-contained static page shipped as package data
(``fastworkflow/run_chatbot/static/index.html``).
"""
