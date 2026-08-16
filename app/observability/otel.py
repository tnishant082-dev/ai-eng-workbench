"""OpenTelemetry hooks — no-op without collector; optional console export."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _console_enabled() -> bool:
    return os.getenv("AIWB_OTEL_CONSOLE", "").lower() in ("1", "true", "yes")


def _otel_available():
    try:
        from opentelemetry import trace  # noqa: F401

        return True
    except Exception:
        return False


@contextmanager
def start_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Start an OTel span if SDK present; else no-op. Console log if AIWB_OTEL_CONSOLE=1."""
    attrs = attributes or {}
    if _console_enabled():
        print(f"[otel] start span={name} attrs={attrs}")
    if not _otel_available():
        span = _NoopSpan()
        try:
            yield span
        finally:
            if _console_enabled():
                print(f"[otel] end span={name}")
        return

    from opentelemetry import trace

    tracer = trace.get_tracer("ai-eng-workbench")
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            try:
                span.set_attribute(k, v)
            except Exception:
                pass
        try:
            yield span
        finally:
            if _console_enabled():
                print(f"[otel] end span={name}")


def configure_otel_console_exporter() -> str:
    """Best-effort console exporter setup. Returns status string."""
    if not _otel_available():
        return "opentelemetry_not_installed"
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        return "console_exporter_configured"
    except Exception as exc:  # noqa: BLE001
        return f"otel_configure_failed: {exc}"
