"""Live simulation WebSocket.

Protocol (client → server):

    {"type": "start", "interval_ms": 700}
    {"type": "pause"}
    {"type": "step"}
    {"type": "reset", "seed": 123, "keep_policy_learning": false}
    {"type": "speed", "interval_ms": 250}

Server → client events: ``snapshot``, ``step``, ``status``, ``terminated``, ``error``.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..db import session_scope
from ..services import hub
from ..services.run_service import RunNotFound, RunService, run_lock

router = APIRouter(tags=["ws"])

MIN_INTERVAL_S = 0.05
MAX_INTERVAL_S = 5.0


def _clamp_interval(interval_ms: float | int | None, fallback: float) -> float:
    if interval_ms is None:
        return fallback
    return max(MIN_INTERVAL_S, min(float(interval_ms) / 1000.0, MAX_INTERVAL_S))


@router.websocket("/ws/runs/{run_id}")
async def run_socket(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()

    try:
        with session_scope() as session:
            detail = RunService(session).detail(run_id)
    except RunNotFound:
        await websocket.send_json({"type": "error", "detail": f"Run '{run_id}' not found"})
        await websocket.close()
        return

    # All sends funnel through this lock: the playback loop and the spectator forwarder both
    # write to the same socket, and interleaved frames would corrupt the stream.
    send_lock = asyncio.Lock()

    async def send(message: dict) -> None:
        async with send_lock:
            await websocket.send_json(message)

    await send({"type": "snapshot", "run": detail})

    queue = hub.subscribe(run_id)
    forwarder = asyncio.create_task(_forward(queue, send))

    playing = False
    interval = 0.7
    lock = run_lock(run_id)

    try:
        while True:
            command = None
            if playing:
                with contextlib.suppress(asyncio.TimeoutError):
                    command = await asyncio.wait_for(websocket.receive_json(), timeout=interval)
            else:
                command = await websocket.receive_json()

            if command:
                kind = command.get("type")
                if kind == "start":
                    playing = True
                    interval = _clamp_interval(command.get("interval_ms"), interval)
                    await _emit_status(send, run_id, "running", playing, interval)
                    continue
                if kind == "pause":
                    playing = False
                    await _emit_status(send, run_id, "paused", playing, interval)
                    continue
                if kind == "speed":
                    interval = _clamp_interval(command.get("interval_ms"), interval)
                    await _emit_status(send, run_id, None, playing, interval)
                    continue
                if kind == "reset":
                    playing = False
                    async with lock:
                        with session_scope() as session:
                            detail = RunService(session).reset(
                                run_id,
                                seed=command.get("seed"),
                                keep_policy_learning=bool(
                                    command.get("keep_policy_learning", False)
                                ),
                            )
                    await send({"type": "snapshot", "run": detail})
                    continue
                if kind == "close":
                    break
                if kind != "step":
                    await send({"type": "error", "detail": f"Unknown command '{kind}'"})
                    continue

            # Either an explicit {"type":"step"} or the playback timer fired.
            async with lock:
                with session_scope() as session:
                    svc = RunService(session)
                    engine = svc.engine_for(run_id)
                    if engine.done:
                        playing = False
                        await send(
                            {
                                "type": "terminated",
                                "run": svc.detail(run_id),
                                "reason": engine.state.termination_reason,
                            }
                        )
                        continue
                    result = svc.step(run_id)
                    detail = svc.detail(run_id)

            await send({"type": "step", "step": result, "run": detail})
            if result["done"]:
                playing = False
                await send(
                    {
                        "type": "terminated",
                        "run": detail,
                        "reason": result["termination_reason"],
                    }
                )

    except WebSocketDisconnect:
        return
    except Exception as exc:  # surface engine errors to the client instead of dropping
        with contextlib.suppress(Exception):
            await send({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
    finally:
        forwarder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await forwarder
        hub.unsubscribe(run_id, queue)
        with contextlib.suppress(Exception):
            await websocket.close()


async def _forward(queue: asyncio.Queue, send) -> None:
    """Relay hub events (live steps reported over REST) to this socket."""
    while True:
        message = await queue.get()
        with contextlib.suppress(Exception):
            await send(message)


async def _emit_status(
    send, run_id: str, status: str | None, playing: bool, interval: float
) -> None:
    if status:
        with session_scope() as session:
            RunService(session).set_status(run_id, status)
    await send(
        {
            "type": "status",
            "playing": playing,
            "interval_ms": int(interval * 1000),
            "status": status,
        }
    )
