"""Compute resource and execution logging for SymRM."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional
import torch


class ComputeTracker:
    def __init__(
        self,
        log_path: str | Path,
        run_id: str,
        model: str,
        condition: str,
        seed: int,
    ) -> None:
        self.log_path = Path(log_path)
        self.run_id = run_id
        self.model = model
        self.condition = condition
        self.seed = seed
        self.start_time: Optional[float] = None
        self.total_tokens: int = 0

    def start(self) -> None:
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record_tokens(self, count: int) -> None:
        self.total_tokens += count

    def stop(self) -> dict[str, float | str | int]:
        if self.start_time is None:
            raise RuntimeError("Tracker was never started.")
        wall_clock_sec = time.perf_counter() - self.start_time
        wall_clock_min = round(wall_clock_sec / 60.0, 3)
        gpu_hours = round(wall_clock_sec / 3600.0, 5)

        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

        tokens_per_sec = round(self.total_tokens / max(wall_clock_sec, 1e-6), 2)

        record = {
            "run_id": self.run_id,
            "model": self.model,
            "condition": self.condition,
            "seed": self.seed,
            "peak_vram_gb": peak_vram_gb,
            "tokens_per_sec": tokens_per_sec,
            "wall_clock_min": wall_clock_min,
            "gpu_hours": gpu_hours,
        }

        self._append_to_csv(record)
        return record

    def _append_to_csv(self, record: dict[str, float | str | int]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists() and self.log_path.stat().st_size > 0

        fieldnames = [
            "run_id",
            "model",
            "condition",
            "seed",
            "peak_vram_gb",
            "tokens_per_sec",
            "wall_clock_min",
            "gpu_hours",
        ]

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
