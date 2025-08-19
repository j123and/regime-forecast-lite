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
                for row in reader:
                    tick: dict[str, Any] = {
                        "timestamp": row[self.ts_col],
                        "x": float(row[self.y_col]),
                        "covariates": {},
                    }
                    for c in self.covar_cols:
                        if c in row and row[c] != "":
                            tick["covariates"][c] = float(row[c])
                    yield tick
        else:
            import pandas as pd  # type: ignore[import-not-found]
            df = pd.read_parquet(self.path)
            n = len(df)
            for start in range(0, n, self.batch_size):
                chunk = df.iloc[start : start + self.batch_size]
                for _, r in chunk.iterrows():
                    cov = {c: float(r[c]) for c in self.covar_cols if c in chunk.columns}
                    yield {
                        "timestamp": str(r[self.ts_col]),
                        "x": float(r[self.y_col]),
                        "covariates": cov,
                    }
