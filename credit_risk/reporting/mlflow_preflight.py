"""Fail-fast MLflow reachability check for long studies.

`train_model` opens an MLflow run for every step even with register=False, so
an unreachable tracking server kills a study that may already be hours in.
`MLflowRegistry` swallows the connection error at init and only raises later
inside `start_run`, which is why this check has to be explicit and up front.

Shared by the ablation (T1.2) and cross-validation (T1.4) pipelines.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)

_PREFLIGHT_TCP_TIMEOUT_SEC = 3.0


def _check_tcp_reachable(uri: str) -> None:
    """Fail immediately if an http(s) tracking URI refuses a TCP connection.

    Needed because MlflowClient retries a dead host seven times with backoff
    (over two minutes measured), and the point of a preflight is to stop *now*.
    Non-HTTP backends (file:, sqlite:, databricks:) have nothing to probe.
    """
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection(
            (parsed.hostname, port), timeout=_PREFLIGHT_TCP_TIMEOUT_SEC
        ):
            return
    except OSError as e:
        raise RuntimeError(
            f"MLflow at {uri} is not reachable: {type(e).__name__}: {e}\n"
            "Every step of this study opens an MLflow run, so it would die on "
            "the first one.\n"
            "Start the tracking server before launching, or point at a local "
            "file store:\n"
            "  export MLFLOW_TRACKING_URI=file:./mlruns"
        ) from e


def require_mlflow(what: str = "study") -> str:
    """Return the tracking URI, or raise with an actionable message.

    Requires MLFLOW_TRACKING_URI to be set explicitly: a multi-step study must
    not scatter runs into whatever default happens to be configured.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        cfg_uri = get_config().get("mlflow", {}).get("tracking_uri")
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set.\n"
            f"Every step of this {what} opens an MLflow run, so the destination "
            "must be explicit — otherwise the runs land somewhere unintended.\n"
            f"  config default would be: {cfg_uri}\n"
            "  export MLFLOW_TRACKING_URI=http://localhost:5000\n"
            "  # or, for a local file store:\n"
            "  export MLFLOW_TRACKING_URI=file:./mlruns"
        )

    _check_tcp_reachable(uri)

    # Cap MLflow's own retries for the API probe too, so a server that accepts
    # connections but errors out still fails in seconds rather than minutes.
    saved = {k: os.environ.get(k) for k in
             ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_TIMEOUT")}
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    try:
        from mlflow import MlflowClient
        MlflowClient(tracking_uri=uri).search_experiments(max_results=1)
    except Exception as e:
        raise RuntimeError(
            f"MLflow at {uri} accepted a connection but the API call failed: "
            f"{type(e).__name__}: {e}\n"
            "Check the tracking server's health before launching the study."
        ) from e
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    log.info("mlflow_preflight_ok", uri=uri)
    return uri
