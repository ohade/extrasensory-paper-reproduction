"""Append-only, process-safe experiment ledger and immutable run directories."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_dir = root / "runs"
        self.ledger_path = root / "ledger.jsonl"

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append(self, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            **record,
        }
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.ledger_path.exists():
            return
        for line in self.ledger_path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)
