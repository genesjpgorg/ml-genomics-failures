"""Hosted MCP endpoint on Modal: https://<workspace>--ml-genomics-failures.modal.run/mcp

Deploy from the repo root: uvx modal deploy deploy/modal_app.py
Scales to zero when idle (cold start a few seconds); stateless, so any number of replicas can serve.
"""

from pathlib import Path

import modal

SRC = Path(__file__).resolve().parents[1] / "src" / "ml_genomics_failures"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("mcp>=2.3,<3", "numpy>=1.26", "uvicorn>=0.30")
    .add_local_dir(SRC, "/root/ml_genomics_failures")
)
app = modal.App("ml-genomics-failures", image=image)


@app.function(cpu=1.0, memory=2048, timeout=120, scaledown_window=300)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="ml-genomics-failures")
def web():
    from ml_genomics_failures import server

    return server.http_app(public=True)
