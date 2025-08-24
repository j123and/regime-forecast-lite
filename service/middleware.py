# service/middleware.py
import time

from starlette.middleware.base import BaseHTTPMiddleware


class ServiceTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000.0
        # if JSON response, inject field; otherwise add header
        try:
            body = b"".join([chunk async for chunk in response.body_iterator])
            # rebuild Response with injected JSON "latency_ms": {"service_ms": ms}
            import json
            data = json.loads(body)
            data.setdefault("latency_ms", {})["service_ms"] = round(ms, 3)
            from starlette.responses import JSONResponse
            return JSONResponse(data, status_code=response.status_code)
        except Exception:
            response.headers["X-Service-MS"] = f"{ms:.3f}"
            return response
