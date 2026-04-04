from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import orjson


class JsonlLogger:
    def __init__(self, file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def log(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "ts_unix_ms": int(time.time() * 1000),
            "event": event,
            **payload,
        }
        with self._path.open("ab") as f:
            f.write(orjson.dumps(row))
            f.write(b"\n")
