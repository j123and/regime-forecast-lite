from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_metrics(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    # backtest.cli prints {"metrics": {...}, "n_points": N}
    return data


def _fmt(v: Any, fmt: str) -> str:
    try:
        return format(float(v), fmt)
    except Exception:
        return str(v)


def _section(title: str, data: dict[str, Any]) -> str:
    m = data.get("metrics", {})
    n = data.get("n_points", 0)

    lines = [f"### {title}", ""]
    lines.append(f"- Points: **{n}**")
    if "mae" in m:
        lines.append(f"- MAE: **{_fmt(m['mae'], '.4f')}**")
    if "rmse" in m:
        lines.append(f"- RMSE: **{_fmt(m['rmse'], '.4f')}**")
    if "smape" in m:
        lines.append(f"- sMAPE: **{_fmt(m['smape'], '.2f')}%**")
    if "coverage" in m:
        lines.append(f"- Coverage (1-α): **{_fmt(m['coverage'], '.3f')}**")
    if "latency_p50_ms" in m and "latency_p95_ms" in m:
        lines.append(
            f"- Latency p50/p95: **{_fmt(m['latency_p50_ms'], '.1f')} / {_fmt(m['latency_p95_ms'], '.1f')} ms**"
        )

    # CP metrics if present
    cp_keys = [
        "cp_precision",
        "cp_recall",
        "cp_delay_mean",
        "cp_delay_p95",
        "cp_chatter_per_1000",
        "cp_false_alarm_rate",
    ]
    if any(k in m for k in cp_keys):
        lines.append("- Change-point metrics:")
        if "cp_precision" in m:
            lines.append(f"  - Precision: **{_fmt(m['cp_precision'], '.3f')}**")
        if "cp_recall" in m:
            lines.append(f"  - Recall: **{_fmt(m['cp_recall'], '.3f')}**")
        if "cp_delay_mean" in m:
            lines.append(f"  - Mean delay (ticks): **{_fmt(m['cp_delay_mean'], '.2f')}**")
        if "cp_delay_p95" in m:
            lines.append(f"  - p95 delay (ticks): **{_fmt(m['cp_delay_p95'], '.0f')}**")
        if "cp_chatter_per_1000" in m:
            lines.append(f"  - Chatter (/1000): **{_fmt(m['cp_chatter_per_1000'], '.1f')}**")
        if "cp_false_alarm_rate" in m:
            lines.append(f"  - False alarm rate: **{_fmt(m['cp_false_alarm_rate'], '.3f')}**")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parts: list[str] = []
    sim = _load_metrics("artifacts/metrics_sim.json")
    if sim:
        parts.append(_section("Backtest — Simulated Data", sim))
        parts.append("![Simulated Backtest](artifacts/plot_sim.png)\n")

    aapl = _load_metrics("artifacts/metrics_aapl.json")
    if aapl:
        parts.append(_section("Backtest — AAPL 1h Log Returns", aapl))
        parts.append("![AAPL Backtest](artifacts/plot_aapl.png)\n")

    # Service smoke (optional for README)
    svc_path = Path("artifacts/service_smoke.json")
    if svc_path.exists():
        parts.append("### Service Smoke Test")
        parts.append("")
        parts.append("Example `/predict` ➜ `/truth` roundtrip (shortened):")
        try:
            svc = json.loads(svc_path.read_text(encoding="utf-8"))
            pred = svc.get("predict", {})
            truth = svc.get("truth", {})
            parts.append("")
            parts.append(f"- `prediction_id`: `{pred.get('prediction_id', '')}`")
            parts.append(
                f"- regime: `{pred.get('regime', '')}`  | score: `{pred.get('score', 0):.3f}`"
            )
            parts.append(
                f"- truth status: `{truth.get('status', '')}`  | idempotent: `{truth.get('idempotent', '')}`"
            )
        except Exception:
            parts.append("- (Could not parse `artifacts/service_smoke.json`)")
        parts.append("")

    out = "\n".join(parts).strip() + "\n"
    Path("artifacts/readme_metrics.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
