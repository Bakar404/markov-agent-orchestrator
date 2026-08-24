"""In-process pub/sub so REST-driven live steps reach WebSocket spectators.

Sim mode advances through the WebSocket itself, so the socket that issued the step also emits
it. Live mode advances through ``POST /live/report``, which has no socket, so anyone watching
the arena would sit on a stale board. This hub bridges the two.

Publishing happens from FastAPI's threadpool (the endpoints are sync ``def``), while the queues
are consumed on the event loop, so delivery hops back via ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

_subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
_loop: asyncio.AbstractEventLoop | None = None

MAX_PENDING = 64


def subscribe(run_id: str) -> asyncio.Queue:
    """Register a listener. Must be called from the event loop."""
    global _loop
    _loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_PENDING)
    _subscribers[run_id].add(queue)
    return queue


def unsubscribe(run_id: str, queue: asyncio.Queue) -> None:
    listeners = _subscribers.get(run_id)
    if not listeners:
        return
    listeners.discard(queue)
    if not listeners:
        _subscribers.pop(run_id, None)


def publish(run_id: str, message: dict) -> None:
    """Fan a message out to every listener. Safe to call from a worker thread."""
    listeners = _subscribers.get(run_id)
    if not listeners or _loop is None or _loop.is_closed():
        return

    for queue in list(listeners):
        _loop.call_soon_threadsafe(_offer, queue, message)


def _offer(queue: asyncio.Queue, message: dict) -> None:
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        # A spectator that cannot keep up loses frames rather than stalling the run.
        pass


def listener_count(run_id: str) -> int:
    return len(_subscribers.get(run_id, ()))
