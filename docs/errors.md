

# API Errors

We return JSON errors with a stable shape:

```json
{
  "error": "short_code",
  "detail": "human-readable cause",
  "status": 401
}
````

## 401 Unauthorized

Missing or invalid token.

**When:**

* `SERVICE_TOKEN` is set server-side and the `Authorization` header is missing or not `Bearer <token>`.

**Example:**

```json
{
  "error": "unauthorized",
  "detail": "Missing or invalid Bearer token",
  "status": 401
}
```

## 403 Forbidden

Token presented but not permitted (e.g., token list configured and this token not in allowlist).

**Example:**

```json
{
  "error": "forbidden",
  "detail": "Token not permitted for this resource",
  "status": 403
}
```

## 409 Conflict

Conflict with existing state; idempotency or duplicate truth.

**When:**

* `/truth` for the same (`series_id`, `target_timestamp`) or `prediction_id` was already accepted.
* A `prediction_id` reused for a different payload.

**Example (duplicate truth):**

```json
{
  "error": "conflict",
  "detail": "truth already recorded for prediction_id=6c3f... or (series_id=X, target_timestamp=...)",
  "status": 409
}
```

## 422 Unprocessable Entity

Malformed body or schema mismatch (Pydantic validation failed).

**When:**

* Types wrong (e.g., `x` is a string).
* Missing required fields (`timestamp`, `x` for /predict; `prediction_id` or (`series_id`,`target_timestamp`) for /truth).
* Bad timestamp format (must be ISO-8601 `Z` or epoch seconds as string).

**Example:**

```json
{
  "error": "validation_error",
  "detail": "Field 'timestamp' must be ISO-8601 Z or epoch string; got '2024-01-02 10:00:00+02:00'",
  "status": 422
}
```

## 429 Too Many Requests

Rate limit exceeded.

**When:**

* Token/IP exceeds `RATE_LIMIT_RPS` or `RATE_LIMIT_BURST`.

**Example:**

```json
{
  "error": "rate_limited",
  "detail": "Exceeded 100 req/s (burst 200) for token=***, try again later",
  "status": 429,
  "retry_after_s": 1
}
```


If your app currently returns FastAPI’s default error JSON for some cases, you can still document these shapes and gradually standardize by raising `HTTPException(status_code, detail=<dict>)` where `detail` is already your structured object.

---
