"""Run service: engine lifecycle, persistence and aggregation.

Engines are cached in-process for speed but the database is always the source of truth: every
step writes a state snapshot, a trace and the emitted messages, and updates the run row with
the serialized engine snapshot. A cache miss simply rehydrates the engine from that snapshot.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections import defaultdict

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..models import Message, Paper, Run, StateSnapshot, Trace
from ..orchestration.agents import AGENTS, agent_catalog
from ..orchestration.engine import OrchestrationEngine, RunConfig, StepResult
from ..orchestration.live import report_from_response
from ..orchestration.strategies import get_strategy
from ..schemas import RunCreate

_engines: dict[str, OrchestrationEngine] = {}
_locks: dict[str, asyncio.Lock] = {}


def run_lock(run_id: str) -> asyncio.Lock:
    lock = _locks.get(run_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[run_id] = lock
    return lock


class RunNotFound(LookupError):
    pass


class RunService:
    def __init__(self, session: Session) -> None:
        self.session = session

    # -------------------------------------------------------------- create
    def create(self, payload: RunCreate) -> dict:
        seed = payload.seed if payload.seed is not None else random.randrange(1, 2**31 - 1)

        policy = payload.policy
        policy_options = dict(payload.policy_options)
        arm = payload.arm
        if payload.strategy:
            strategy = get_strategy(payload.strategy)
            policy = strategy.policy
            policy_options = {**(strategy.policy_options or {}), **policy_options}
            arm = arm or strategy.id

        config = RunConfig(
            task=payload.task,
            policy=policy,
            seed=seed,
            task_complexity=payload.task_complexity,
            budget_usd=payload.budget_usd,
            latency_budget_ms=payload.latency_budget_ms,
            max_steps=payload.max_steps,
            belief_dim=payload.belief_dim,
            stochasticity=payload.stochasticity,
            confidence_target=payload.confidence_target,
            verification_target=payload.verification_target,
            min_steps_before_terminate=payload.min_steps_before_terminate,
            mode=payload.mode,
            hypotheses=list(payload.hypotheses),
            task_shape=dict(payload.task_shape),
            default_model=payload.default_model,
            agent_models=dict(payload.agent_models),
            experiment=payload.experiment,
            arm=arm,
            policy_options=policy_options,
        )
        engine = OrchestrationEngine(config)
        snapshot = engine.snapshot()

        run = Run(
            task=payload.task,
            policy=policy,
            status="created",
            seed=seed,
            config=snapshot["config"],
            current_state=snapshot["state"],
            initial_state=snapshot["initial_state"],
            policy_state=snapshot,
        )
        self.session.add(run)
        self.session.flush()

        self.session.add(
            StateSnapshot(
                run_id=run.id,
                step=0,
                entropy=engine.state.entropy,
                confidence=engine.state.confidence,
                payload=snapshot["state"],
            )
        )
        self.session.commit()
        _engines[run.id] = engine
        return self.detail(run.id)

    # -------------------------------------------------------------- engine
    def engine_for(self, run_id: str) -> OrchestrationEngine:
        engine = _engines.get(run_id)
        if engine is not None:
            return engine
        run = self._run(run_id)
        engine = OrchestrationEngine.restore(run.policy_state or {})
        _engines[run_id] = engine
        return engine

    def _run(self, run_id: str) -> Run:
        run = self.session.get(Run, run_id)
        if run is None:
            raise RunNotFound(f"Run '{run_id}' not found")
        return run

    # ---------------------------------------------------------------- step
    def step(self, run_id: str) -> dict:
        run = self._run(run_id)
        engine = self.engine_for(run_id)
        if engine.done:
            raise ValueError("Run has already terminated")
        if engine.is_live:
            # Sampling a step here would fabricate an outcome inside a real episode.
            raise ValueError("Live runs advance through /live/open and /live/report, not /step")

        result = engine.step()
        self._persist_step(run, engine, result)
        return result.to_dict()

    def step_many(self, run_id: str, steps: int) -> list[dict]:
        results: list[dict] = []
        for _ in range(steps):
            engine = self.engine_for(run_id)
            if engine.done:
                break
            results.append(self.step(run_id))
        return results

    # ----------------------------------------------------------- live mode
    def live_open(self, run_id: str) -> dict:
        """Ask the policy who acts next and return their brief. State does not advance."""
        run = self._run(run_id)
        engine = self.engine_for(run_id)
        if not engine.is_live:
            raise ValueError("Run is not in live mode")
        if engine.done:
            raise ValueError("Run has already terminated")

        pending = engine.open_step(uuid.uuid4().hex)
        pending.run_id = run_id
        if run.status != "running":
            run.status = "running"
            self.session.commit()
        return pending.to_dict()

    def live_report(
        self,
        run_id: str,
        token: str,
        reports: list,
        hypotheses: list[str] | None = None,
    ) -> dict:
        """Fold real agent output into the episode and advance one step."""
        run = self._run(run_id)
        engine = self.engine_for(run_id)
        if not engine.is_live:
            raise ValueError("Run is not in live mode")

        if hypotheses and not engine.config.hypotheses:
            if len(hypotheses) != engine.config.belief_dim:
                raise ValueError(
                    f"expected {engine.config.belief_dim} hypotheses, got {len(hypotheses)}"
                )
            engine.config.hypotheses = list(hypotheses)

        # engine.state is still the pre-step state until close_step runs, which is exactly the
        # state the briefs were built from.
        unknown = sorted({item.agent_id for item in reports} - set(AGENTS))
        if unknown:
            raise ValueError(f"unknown agent id(s): {', '.join(unknown)}")

        self._reject_replayed_responses(run, reports)

        built = [
            report_from_response(
                AGENTS[item.agent_id],
                engine.state,
                item.model_dump(),
                belief_dim=engine.config.belief_dim,
            )
            for item in reports
        ]

        result = engine.close_step(token, built)
        self._persist_step(run, engine, result)
        return result.to_dict()

    def live_abandon(self, run_id: str) -> None:
        self.engine_for(run_id).abandon_step()

    def _reject_replayed_responses(self, run: Run, reports: list) -> None:
        """Refuse work this run has already submitted.

        Re-sending earlier output is not another step, and once persisted nothing downstream can
        tell diligence from copy-paste. Only summaries are stored, so that is what is compared;
        responses are checked within the batch, where both are in hand.
        """
        seen = set(
            self.session.scalars(
                select(Message.content).where(
                    Message.run_id == run.id, Message.kind != "handoff"
                )
            ).all()
        )

        batch_responses: set[str] = set()
        for item in reports:
            body = (item.response or "").strip()
            if body in batch_responses:
                raise ValueError(
                    f"agent '{item.agent_id}' submitted the same response as another agent in "
                    "this step"
                )
            batch_responses.add(body)

            # Mirrors how live.py derives a summary when the caller omits one.
            summary = (item.summary or "").strip()
            if not summary:
                summary = body.splitlines()[0][:160] if body else ""
            if summary and summary in seen:
                raise ValueError(
                    f"agent '{item.agent_id}' reported work this run already recorded "
                    f"({summary[:60]!r}). Repeating earlier output is not another step."
                )
            seen.add(summary)

    def _persist_step(self, run: Run, engine: OrchestrationEngine, result: StepResult) -> None:
        snapshot = engine.snapshot()

        state_row = StateSnapshot(
            run_id=run.id,
            step=result.step,
            entropy=result.entropy_after,
            confidence=result.confidence,
            payload=result.state,
        )
        self.session.add(state_row)
        self.session.flush()

        prev_state_row = self.session.scalar(
            select(StateSnapshot).where(
                StateSnapshot.run_id == run.id, StateSnapshot.step == result.step - 1
            )
        )

        self.session.add(
            Trace(
                run_id=run.id,
                step=result.step,
                state_id=state_row.id,
                prev_state_id=prev_state_row.id if prev_state_row else None,
                action=result.action,
                agents=result.agents,
                action_probability=result.action_probability,
                transition_probability=result.transition_probability,
                action_distribution=result.action_distribution,
                outcome=result.outcome,
                confidence=result.confidence,
                entropy_before=result.entropy_before,
                entropy_after=result.entropy_after,
                information_gain=result.information_gain,
                reward=result.reward,
                cumulative_reward=result.cumulative_reward,
                reward_breakdown=result.reward_breakdown,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                tokens=result.tokens,
                notes=result.notes,
            )
        )

        for message in result.messages:
            self.session.add(
                Message(
                    run_id=run.id,
                    step=message["step"],
                    sender=message["sender"],
                    receiver=message["receiver"],
                    kind=message["kind"],
                    content=message["content"],
                    weight=message["weight"],
                )
            )

        run.step_count = result.step
        run.cumulative_reward = result.cumulative_reward
        run.total_cost = engine.total_cost
        run.total_latency_ms = engine.total_latency_ms
        run.total_tokens = engine.total_tokens
        run.terminated = result.done
        run.termination_reason = result.termination_reason
        run.current_state = result.state
        run.policy_state = snapshot
        run.status = "completed" if result.done else "running"
        self.session.commit()

    # --------------------------------------------------------------- reset
    def reset(self, run_id: str, *, seed: int | None = None, keep_policy_learning: bool = False) -> dict:
        run = self._run(run_id)
        previous = _engines.get(run_id)
        config = RunConfig.from_dict(run.config)
        config.seed = seed if seed is not None else random.randrange(1, 2**31 - 1)

        engine = OrchestrationEngine(config)
        if keep_policy_learning and previous is not None:
            engine.policy.load_state_dict(previous.policy.state_dict())

        self.session.execute(delete(Trace).where(Trace.run_id == run.id))
        self.session.execute(delete(Message).where(Message.run_id == run.id))
        self.session.execute(delete(StateSnapshot).where(StateSnapshot.run_id == run.id))

        snapshot = engine.snapshot()
        run.seed = config.seed
        run.status = "created"
        run.step_count = 0
        run.cumulative_reward = 0.0
        run.total_cost = 0.0
        run.total_latency_ms = 0.0
        run.total_tokens = 0
        run.terminated = False
        run.termination_reason = None
        run.config = snapshot["config"]
        run.current_state = snapshot["state"]
        run.initial_state = snapshot["initial_state"]
        run.policy_state = snapshot

        self.session.add(
            StateSnapshot(
                run_id=run.id,
                step=0,
                entropy=engine.state.entropy,
                confidence=engine.state.confidence,
                payload=snapshot["state"],
            )
        )
        self.session.commit()
        _engines[run_id] = engine
        return self.detail(run_id)

    def set_status(self, run_id: str, status: str) -> dict:
        run = self._run(run_id)
        if not run.terminated:
            run.status = status
        self.session.commit()
        return self.detail(run_id)

    def delete(self, run_id: str) -> None:
        run = self._run(run_id)
        self.session.delete(run)
        self.session.commit()
        _engines.pop(run_id, None)
        _locks.pop(run_id, None)

    # ------------------------------------------------------------ read side
    def list_runs(self, *, limit: int = 50) -> list[dict]:
        runs = self.session.scalars(
            select(Run).order_by(Run.created_at.desc()).limit(limit)
        ).unique()
        return [self._summary(run) for run in runs]

    def detail(self, run_id: str) -> dict:
        run = self._run(run_id)
        engine = self.engine_for(run_id)
        payload = self._summary(run)
        payload.update(
            {
                "config": run.config,
                "state": run.current_state,
                "initial_state": run.initial_state,
                "preview": engine.preview(),
                "agents": agent_catalog(),
            }
        )
        return payload

    def traces(self, run_id: str, *, limit: int = 500) -> list[dict]:
        self._run(run_id)
        rows = self.session.scalars(
            select(Trace).where(Trace.run_id == run_id).order_by(Trace.step).limit(limit)
        ).unique()
        return [
            {
                "id": row.id,
                "run_id": row.run_id,
                "step": row.step,
                "state_id": row.state_id,
                "prev_state_id": row.prev_state_id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "action": row.action,
                "agents": list(row.agents or []),
                "action_probability": row.action_probability,
                "transition_probability": row.transition_probability,
                "action_distribution": row.action_distribution,
                "outcome": row.outcome,
                "confidence": row.confidence,
                "entropy_before": row.entropy_before,
                "entropy_after": row.entropy_after,
                "information_gain": row.information_gain,
                "reward": row.reward,
                "cumulative_reward": row.cumulative_reward,
                "reward_breakdown": row.reward_breakdown,
                "latency_ms": row.latency_ms,
                "cost_usd": row.cost_usd,
                "tokens": row.tokens,
                "notes": row.notes,
            }
            for row in rows
        ]

    def messages(self, run_id: str, *, limit: int = 1000) -> list[dict]:
        self._run(run_id)
        rows = self.session.scalars(
            select(Message).where(Message.run_id == run_id).order_by(Message.step).limit(limit)
        ).unique()
        return [
            {
                "id": row.id,
                "step": row.step,
                "sender": row.sender,
                "receiver": row.receiver,
                "kind": row.kind,
                "content": row.content,
                "weight": row.weight,
            }
            for row in rows
        ]

    def states(self, run_id: str, *, limit: int = 500) -> list[dict]:
        self._run(run_id)
        rows = self.session.scalars(
            select(StateSnapshot)
            .where(StateSnapshot.run_id == run_id)
            .order_by(StateSnapshot.step)
            .limit(limit)
        ).unique()
        return [
            {
                "id": row.id,
                "step": row.step,
                "entropy": row.entropy,
                "confidence": row.confidence,
                "payload": row.payload,
            }
            for row in rows
        ]

    def metrics(self, run_id: str) -> dict:
        run = self._run(run_id)
        traces = self.traces(run_id)

        per_agent: dict[str, dict] = defaultdict(
            lambda: {
                "reward": 0.0,
                "cost": 0.0,
                "latency_ms": 0.0,
                "tokens": 0,
                "invocations": 0,
                "information_gain": 0.0,
            }
        )
        for trace in traces:
            breakdown = trace["reward_breakdown"] or {}
            shares = breakdown.get("per_agent") or {}
            count = max(len(trace["agents"]), 1)
            for agent_id in trace["agents"]:
                bucket = per_agent[agent_id]
                bucket["reward"] += float(shares.get(agent_id, trace["reward"] / count))
                bucket["cost"] += trace["cost_usd"] / count
                bucket["latency_ms"] += trace["latency_ms"] / count
                bucket["tokens"] += trace["tokens"] // count
                bucket["invocations"] += 1
                bucket["information_gain"] += trace["information_gain"] / count

        state = run.current_state or {}
        cumulative_reward = run.cumulative_reward
        totals = {
            "cumulative_reward": cumulative_reward,
            "total_cost": run.total_cost,
            "total_latency_ms": run.total_latency_ms,
            "total_tokens": run.total_tokens,
            "steps": run.step_count,
            "cost_efficiency": cumulative_reward / run.total_cost if run.total_cost > 0 else 0.0,
            "reward_per_step": cumulative_reward / run.step_count if run.step_count else 0.0,
            "tokens_per_step": run.total_tokens / run.step_count if run.step_count else 0.0,
            "total_information_gain": sum(t["information_gain"] for t in traces),
            "quality": state.get("quality", 0.0),
            "verification_score": state.get("verification_score", 0.0),
            "confidence": state.get("confidence", 0.0),
            "entropy": state.get("entropy", 0.0),
            "memory_coverage": state.get("memory_coverage", 0.0),
        }

        reward_terms = defaultdict(float)
        for trace in traces:
            for term, value in (trace["reward_breakdown"] or {}).items():
                if term in {"per_agent", "total"}:
                    continue
                reward_terms[term] += float(value)

        return {
            "run_id": run_id,
            "totals": totals,
            "reward_terms": dict(reward_terms),
            "per_agent": [
                {
                    "agent_id": agent_id,
                    "label": AGENTS[agent_id].label if agent_id in AGENTS else agent_id,
                    "color": AGENTS[agent_id].color if agent_id in AGENTS else "#94a3b8",
                    **values,
                    "cost_efficiency": values["reward"] / values["cost"] if values["cost"] > 0 else 0.0,
                }
                for agent_id, values in per_agent.items()
            ],
            "series": [
                {
                    "step": trace["step"],
                    "reward": trace["reward"],
                    "cumulative_reward": trace["cumulative_reward"],
                    "entropy_before": trace["entropy_before"],
                    "entropy_after": trace["entropy_after"],
                    "information_gain": trace["information_gain"],
                    "confidence": trace["confidence"],
                    "cost_usd": trace["cost_usd"],
                    "latency_ms": trace["latency_ms"],
                    "tokens": trace["tokens"],
                    "action": trace["action"],
                    "agents": trace["agents"],
                }
                for trace in traces
            ],
        }

    def interaction_graph(self, run_id: str) -> dict:
        """Aggregate message counts into a weighted agent interaction graph."""
        messages = self.messages(run_id)
        edges: dict[tuple[str, str], dict] = {}
        for message in messages:
            key = (message["sender"], message["receiver"])
            bucket = edges.setdefault(
                key,
                {"source": key[0], "target": key[1], "count": 0, "weight": 0.0, "kinds": set()},
            )
            bucket["count"] += 1
            bucket["weight"] += message["weight"]
            bucket["kinds"].add(message["kind"].split(":")[0])

        return {
            "edges": [
                {
                    "source": bucket["source"],
                    "target": bucket["target"],
                    "count": bucket["count"],
                    "weight": round(bucket["weight"], 4),
                    "mean_weight": round(bucket["weight"] / bucket["count"], 4),
                    "kinds": sorted(bucket["kinds"]),
                }
                for bucket in edges.values()
            ],
            "messages": messages[-120:],
        }

    @staticmethod
    def _summary(run: Run) -> dict:
        state = run.current_state or {}
        return {
            "id": run.id,
            "task": run.task,
            "policy": run.policy,
            "status": run.status,
            "seed": run.seed,
            "step_count": run.step_count,
            "cumulative_reward": run.cumulative_reward,
            "total_cost": run.total_cost,
            "total_latency_ms": run.total_latency_ms,
            "total_tokens": run.total_tokens,
            "terminated": run.terminated,
            "termination_reason": run.termination_reason,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            "confidence": state.get("confidence", 0.0),
            "entropy": state.get("entropy", 0.0),
        }


def library_size(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Paper)) or 0
