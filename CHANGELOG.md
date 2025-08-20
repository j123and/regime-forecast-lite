```markdown
# Changelog

All notable changes to this project will be documented here.

## [0.1.0] - 2025-08-20
### Added
- Keyed, idempotent API: `series_id`, `prediction_id` (UUID), `target_timestamp`.
- `/truth` now references `prediction_id` or (`series_id`, `target_timestamp`); dedupe and out-of-order safe.
- Token auth (`Authorization: Bearer <token>`) and per-IP/per-token rate limits.
- Snapshot/restore of model + conformal buffers; `/snapshot` and `/restore` endpoints.
- Calibration suite: rolling coverage, per-regime coverage; plots and JSON artifacts.
- Benchmarks: p50/p95/p99 and RPS across concurrencies; environment stamp.

### Fixed
- Online conformal coverage computed **without leakage**; now matches target `1−α`.

### Known limitations
- Single pipeline per worker (global lock) serializes `/predict` within a worker. Use multiple workers for throughput; per-series pipelines TBD.
```
