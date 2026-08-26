"""HTTP client for the arena API. Thin on purpose — the arena owns every rule."""

from __future__ import annotations

from typing import Any

import httpx


class ArenaError(RuntimeError):
    """The arena refused something. The message is the arena's own, which is the useful part."""


class Arena:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.base_url = base_url.rstrip("/")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Arena:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, path, **kwargs)
        if response.is_error:
            # The arena's rejections carry the reasoning — a replayed report, an unmeasured
            # cost, a stale token. Surfacing the raw detail is more useful than a status code.
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            raise ArenaError(f"{method} {path} -> {response.status_code}: {detail}")
        if response.status_code == 204:
            return None
        return response.json()

    def create_run(self, **payload: Any) -> dict:
        return self._request("POST", "/api/runs", json=payload)

    def open_step(self, run_id: str) -> dict:
        return self._request("POST", f"/api/runs/{run_id}/live/open")

    def report_step(self, run_id: str, token: str, reports: list[dict]) -> dict:
        return self._request(
            "POST", f"/api/runs/{run_id}/live/report", json={"token": token, "reports": reports}
        )

    def abandon(self, run_id: str) -> None:
        self._request("POST", f"/api/runs/{run_id}/live/abandon")

    def run_detail(self, run_id: str) -> dict:
        return self._request("GET", f"/api/runs/{run_id}")

    def messages(self, run_id: str) -> list[dict]:
        return self._request("GET", f"/api/runs/{run_id}/messages")

    def record_verdict(self, run_id: str, **payload: Any) -> dict:
        return self._request("POST", f"/api/runs/{run_id}/verdict", json=payload)

    def record_pairwise(self, experiment: str, **payload: Any) -> dict:
        return self._request("POST", f"/api/experiments/{experiment}/pairwise", json=payload)

    def comparison(self, experiment: str) -> dict:
        return self._request("GET", f"/api/experiments/{experiment}")
