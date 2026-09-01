from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.logging import setup_logging

logger = setup_logging()

ROLE_RANK = {"viewer": 1, "operator": 2, "admin": 3}
ADMIN_PREFIXES = ("/api/v1/jobs",)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency = (time.perf_counter() - start) * 1000
        response.headers["X-Request-Id"] = request_id
        role = getattr(request.state, "role", None)
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": round(latency, 2),
                "role": role,
            },
        )
        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str, enabled: bool = True, roles_json: str = "{}"):
        super().__init__(app)
        self.api_key = api_key
        self.enabled = enabled
        try:
            self.roles = json.loads(roles_json or "{}")
        except Exception:
            self.roles = {}
        self.roles.setdefault(api_key, "admin")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/" or path.startswith(("/health", "/docs", "/openapi", "/redoc")):
            return await call_next(request)
        if not self.enabled:
            request.state.role = "admin"
            return await call_next(request)
        key = request.headers.get("x-api-key") or request.query_params.get("api_key")
        role = self.roles.get(key) if key else None
        if role is None:
            return JSONResponse({"detail": "Invalid or missing API key (X-API-Key)"}, status_code=401)
        request.state.role = role
        need = "viewer"
        if any(path.startswith(p) for p in ADMIN_PREFIXES):
            need = "admin"
        elif request.method in ("POST", "PUT", "PATCH", "DELETE"):
            need = "operator"
        if ROLE_RANK.get(role, 0) < ROLE_RANK.get(need, 0):
            return JSONResponse({"detail": f"Forbidden: role '{role}' insufficient (need '{need}')"}, status_code=403)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int = 120):
        super().__init__(app)
        self.limit = limit_per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        window.append(now)
        return await call_next(request)
