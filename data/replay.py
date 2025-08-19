from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any


class Replay:
    def __init__(
        self,
        path: str,
        ts_col: str = "timestamp",
        y_col: str = "x",
        covar_cols: list[str] | None = None,
        batch_size: int = 4096,
    ) -> None:
        self.path = path
        self.ts_col = ts_col
        self.y_col = y_col
        self.covar_cols = covar_cols or []
        self.batch_size = int(batch_size)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".csv":
            import csv
            with open(self.path, newline="") as f:
                reader = csv.DictReader(f)
                has_cp = "cp" in (reader.fieldnames or [])
                for row in reader:
                    tick: dict[str, Any] = {
                        "timestamp": row[self.ts_col],
                        "x": float(row[self.y_col]),
                        "covariates": {},
                    }
                    for c in self.covar_cols:
                        if c in row and row[c] != "":
                            tick["covariates"][c] = float(row[c])
                    if has_cp:
                        val = row.get("cp", "")
                        tick["cp"] = 1 if str(val).strip() in ("1", "true", "True") else 0
                    yield tick
        else:
            import pandas as pd
            df = pd.read_parquet(self.path)
            has_cp = "cp" in df.columns
            for _, r in df.iterrows():
                cov = {c: float(r[c]) for c in self.covar_cols if c in df.columns}
                out: dict[str, Any] = {"timestamp": str(r[self.ts_col]), "x": float(r[self.y_col]), "covariates": cov}
                if has_cp:
                    val = r["cp"]
                    out["cp"] = 1 if str(val).strip() in ("1", "true", "True") else 0
                yield out
