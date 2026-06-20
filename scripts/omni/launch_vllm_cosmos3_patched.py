#!/usr/bin/env python3
"""Launch a Cosmos3-registered vLLM OpenAI server with a route-name guard.

Some accelerator environments pair vLLM with a FastAPI/Prometheus stack where
``prometheus_fastapi_instrumentator`` sees an ``_IncludedRouter`` entry and
assumes every route exposes ``.path``. That crashes every completion request
before it reaches vLLM. This launcher keeps the normal vLLM CLI arguments but
patches the Prometheus route-name helper to use the request path directly.
"""

from __future__ import annotations

import runpy


def patch_prometheus_route_name() -> None:
    try:
        import prometheus_fastapi_instrumentator.routing as routing
    except Exception:
        return

    def get_route_name(request):  # type: ignore[no-untyped-def]
        return str(getattr(request, "scope", {}).get("path") or "unknown")

    routing.get_route_name = get_route_name


def main() -> None:
    patch_prometheus_route_name()
    import vllm_cosmos3

    vllm_cosmos3.register()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
