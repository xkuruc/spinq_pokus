"""Bounded non-blocking event handoff and recoverable local journal."""

from __future__ import annotations

import gzip
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from .common import atomic_json, redact, utc_now


class EventRecorder:
    def __init__(self, out: Path, max_events: int = 128):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_events)
        self.count = 0
        self.dropped = 0
        self.writer_error: str | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def record(self, kind: str, payload: dict[str, Any], *, source: str = "decoded_sdk") -> bool:
        if self._closed:
            return False
        event = {"seq": self.count + self.dropped + 1, "received_utc": utc_now(),
                 "received_monotonic_ns": time.monotonic_ns(), "kind": kind,
                 "source": source, "payload": payload}
        try:
            self.queue.put_nowait(event)
            self.count += 1
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _writer(self) -> None:
        try:
            with (self.out / "events.jsonl").open("a", encoding="utf-8") as target:
                while True:
                    item = self.queue.get()
                    if item is None:
                        self.queue.task_done()
                        break
                    target.write(json.dumps(redact(item), ensure_ascii=False,
                                            allow_nan=True, separators=(",", ":"), default=str) + "\n")
                    target.flush()
                    self.queue.task_done()
        except Exception as exc:  # keep callback thread alive and fail the dataset closed
            self.writer_error = type(exc).__name__

    def close(self) -> dict[str, Any]:
        if not self._closed:
            self._closed = True
            try:
                self.queue.put(None, timeout=2)
            except queue.Full:
                self.writer_error = "writer_queue_stalled"
            self._thread.join(timeout=20)
            if self._thread.is_alive():
                self.writer_error = "writer_deadline_exceeded"
            plain = self.out / "events.jsonl"
            if plain.exists() and not self._thread.is_alive():
                temporary = self.out / "events.jsonl.gz.tmp"
                with plain.open("rb") as source, gzip.open(temporary, "wb") as dest:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        dest.write(chunk)
                os.replace(temporary, self.out / "events.jsonl.gz")
            atomic_json(self.out / "recorder_status.json", self.status())
        return self.status()

    def status(self) -> dict[str, Any]:
        return {"events_enqueued": self.count, "events_dropped": self.dropped,
                "writer_error": self.writer_error,
                "complete": self.dropped == 0 and self.writer_error is None}
