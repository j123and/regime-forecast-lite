# service/middleware.py
from __future__ import annotations

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class ServiceTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000.0

        # add headers; do NOT read/modify the body
        response.headers["X-Service-MS"] = f"{ms:.3f}"

        # standard Server-Timing header (useful in browsers and tools)
        # if something else already set it, append our metric
        existing = response.headers.get("Server-Timing")
        our_metric = f'app;dur={ms:.3f}'
        response.headers["Server-Timing"] = f"{existing}, {our_metric}" if existing else our_metric

        return response
