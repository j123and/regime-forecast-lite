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
                cols = reader.fieldnames or []
                # support cp or is_cp
                cp_field = "cp" if "cp" in cols else ("is_cp" if "is_cp" in cols else None)
                for row in reader:
                    tick: dict[str, Any] = {
                        "timestamp": row[self.ts_col],
                        "x": float(row[self.y_col]),
                        "covariates": {},
                    }
                    for c in self.covar_cols:
                        if c in row and row[c] != "":
                            tick["covariates"][c] = float(row[c])
                    if cp_field is not None:
                        val = row.get(cp_field, "")
                        s = str(val).strip()
                        tick["cp"] = 1 if s in ("1", "true", "True") else 0
                    yield tick
        else:
            import pandas as pd

            df = pd.read_parquet(self.path)
            cols = list(df.columns)
            cp_field = "cp" if "cp" in cols else ("is_cp" if "is_cp" in cols else None)
            for _, r in df.iterrows():
                cov = {c: float(r[c]) for c in self.covar_cols if c in df.columns}
                out: dict[str, Any] = {"timestamp": str(r[self.ts_col]), "x": float(r[self.y_col]), "covariates": cov}
                if cp_field is not None:
                    val = r[cp_field]
                    s = str(val).strip()
                    out["cp"] = 1 if s in ("1", "true", "True") else 0
                yield out
