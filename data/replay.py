from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any


def _parse_boolish(val: Any) -> int:
    """Return 1 for truthy markers, else 0."""
    if val is None:
        return 0
    s = str(val).strip().lower()
    return 1 if s in {"1", "true", "t", "yes", "y"} else 0


class Replay:
    """
    Stream historical ticks from CSV or Parquet.

    CSV: uses Python's csv module (streaming, low memory).
    Parquet: tries PyArrow streaming in batches; falls back to pandas if PyArrow
             isn't installed (then the whole file is loaded once).

    Each yielded item has:
      {
        "timestamp": str,
        "x": float,
        "covariates": dict[str, float],
        # optional:
        "cp": 0|1
      }
    """

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

        # CSV
        if ext == ".csv":
            import csv

            with open(self.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                if self.ts_col not in cols or self.y_col not in cols:
                    raise KeyError(
                        f"Missing required columns '{self.ts_col}'/'{self.y_col}' in {self.path}"
                    )

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
                        tick["cp"] = _parse_boolish(row.get(cp_field))
                    yield tick
            return

        # Parquet path
        try:
            import pyarrow.parquet as pq  # no type: ignore

            pf = pq.ParquetFile(self.path)
            schema = pf.schema_arrow
            cols = [f.name for f in schema]

            if self.ts_col not in cols or self.y_col not in cols:
                raise KeyError(
                    f"Missing required columns '{self.ts_col}'/'{self.y_col}' in {self.path}"
                )

            cp_field = "cp" if "cp" in cols else ("is_cp" if "is_cp" in cols else None)

            for batch in pf.iter_batches(batch_size=max(1, self.batch_size)):
                bd = batch.to_pydict()  # dict[str, list[Any]]
                ts_list = bd[self.ts_col]
                y_list = bd[self.y_col]

                cov_present = [c for c in self.covar_cols if c in bd]

                for i in range(len(ts_list)):
                    cov = {c: float(bd[c][i]) for c in cov_present if bd[c][i] is not None}
                    rec: dict[str, Any] = {
                        "timestamp": str(ts_list[i]),
                        "x": float(y_list[i]),
                        "covariates": cov,
                    }
                    if cp_field is not None:
                        rec["cp"] = _parse_boolish(bd[cp_field][i])
                    yield rec
            return
        except ModuleNotFoundError:
            # fall back to pandas (loads entire file)
            import pandas as pd

            df = pd.read_parquet(self.path)
            cols = list(df.columns)
            if self.ts_col not in cols or self.y_col not in cols:
                # explicit exception chaining
                raise KeyError(
                    f"Missing required columns '{self.ts_col}'/'{self.y_col}' in {self.path}"
                ) from None

            cp_field = "cp" if "cp" in cols else ("is_cp" if "is_cp" in cols else None)
            for _, r in df.iterrows():
                cov = {
                    c: float(r[c]) for c in self.covar_cols if c in df.columns and pd.notna(r[c])
                }
                row_out: dict[str, Any] = {
                    "timestamp": str(r[self.ts_col]),
                    "x": float(r[self.y_col]),
                    "covariates": cov,
                }
                if cp_field is not None:
                    row_out["cp"] = _parse_boolish(r[cp_field])
                yield row_out
