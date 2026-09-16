"""Minimal stub of DLC's core.py for headless server use.

The real core.py imports tkinter UI + tensorflow which don't exist here.
All processors only need `update_status`.
"""

import time


def update_status(message: str, name: str) -> None:
    print(f"INFO:dlc:{name}: {message}", flush=True)