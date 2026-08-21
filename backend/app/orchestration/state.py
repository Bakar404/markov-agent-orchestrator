"""The state space S of the orchestration MDP.

The state is fully observable to the policy and fully serializable, so a run can be persisted
to SQLite after every step and rehydrated later without replaying history.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .agents import AGENT_SPECS
from .entropy import belief_entropy, confidence_from_belief, normalize, normalized_entropy

FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "task_complexity",
    "uncertainty",
    "budget_remaining",
    "latency_remaining",
    "confidence",
    "memory_coverage",
    "unresolved_ratio",
    "quality",
    "verification_score",
    "mean_agent_success",
    "duplicate_pressure",
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class AgentHistory:
    """Beta-Bernoulli success model per agent, updated from realized outcomes."""

    alpha: float
    beta: float
    invocations: int = 0
    successes: int = 0
    partials: int = 0
    failures: int = 0
    last_step: int = -1
    cumulative_reward: float = 0.0
    cumulative_cost: float = 0.0
    cumulative_tokens: int = 0
    cumulative_information_gain: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.alpha / max(self.alpha + self.beta, 1e-9)

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "invocations": self.invocations,
            "successes": self.successes,
            "partials": self.partials,
            "failures": self.failures,
            "last_step": self.last_step,
            "cumulative_reward": self.cumulative_reward,
            "cumulative_cost": self.cumulative_cost,
            "cumulative_tokens": self.cumulative_tokens,
            "cumulative_information_gain": self.cumulative_information_gain,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AgentHistory":
        return cls(
            alpha=float(payload.get("alpha", 1.0)),
            beta=float(payload.get("beta", 1.0)),
            invocations=int(payload.get("invocations", 0)),
            successes=int(payload.get("successes", 0)),
            partials=int(payload.get("partials", 0)),
            failures=int(payload.get("failures", 0)),
            last_step=int(payload.get("last_step", -1)),
            cumulative_reward=float(payload.get("cumulative_reward", 0.0)),
            cumulative_cost=float(payload.get("cumulative_cost", 0.0)),
            cumulative_tokens=int(payload.get("cumulative_tokens", 0)),
            cumulative_information_gain=float(payload.get("cumulative_information_gain", 0.0)),
        )


@dataclass
class OrchestratorState:
    """State S. Every field is either observed by the policy or displayed in the UI."""

    step: int = 0
    task_complexity: float = 0.5
    belief: list[float] = field(default_factory=list)
    budget_total_usd: float = 1.0
    budget_spent_usd: float = 0.0
    latency_budget_ms: float = 60_000.0
    latency_consumed_ms: float = 0.0
    tokens_consumed: int = 0
    quality: float = 0.05
    verification_score: float = 0.0
    memory_coverage: float = 0.05
    total_subtasks: int = 5
    unresolved_subtasks: int = 5
    duplicate_pressure: float = 0.0
    agent_history: dict[str, AgentHistory] = field(default_factory=dict)
    last_agents: list[str] = field(default_factory=list)
    invocation_signatures: list[str] = field(default_factory=list)
    terminated: bool = False
    termination_reason: str | None = None
    latent_hypothesis: int = 0

    # ----------------------------------------------------------------- derived
    @property
    def belief_array(self) -> np.ndarray:
        return np.asarray(self.belief, dtype=float)

    @property
    def belief_probabilities(self) -> np.ndarray:
        return normalize(self.belief_array)

    @property
    def entropy(self) -> float:
        return belief_entropy(self.belief_array)

    @property
    def uncertainty(self) -> float:
        return normalized_entropy(self.belief_probabilities)

    @property
    def confidence(self) -> float:
        return confidence_from_belief(self.belief_array)

    @property
    def budget_remaining_usd(self) -> float:
        return max(self.budget_total_usd - self.budget_spent_usd, 0.0)

    @property
    def budget_remaining(self) -> float:
        return float(np.clip(self.budget_remaining_usd / max(self.budget_total_usd, 1e-9), 0.0, 1.0))

    @property
    def latency_remaining(self) -> float:
        remaining = max(self.latency_budget_ms - self.latency_consumed_ms, 0.0)
        return float(np.clip(remaining / max(self.latency_budget_ms, 1e-9), 0.0, 1.0))

    @property
    def unresolved_ratio(self) -> float:
        return float(np.clip(self.unresolved_subtasks / max(self.total_subtasks, 1), 0.0, 1.0))

    @property
    def mean_agent_success(self) -> float:
        if not self.agent_history:
            return 0.5
        return float(np.mean([h.success_rate for h in self.agent_history.values()]))

    def features(self) -> np.ndarray:
        """φ(s) — the context vector consumed by every learning policy."""
        return np.array(
            [
                1.0,
                self.task_complexity,
                self.uncertainty,
                self.budget_remaining,
                self.latency_remaining,
                self.confidence,
                self.memory_coverage,
                self.unresolved_ratio,
                self.quality,
                self.verification_score,
                self.mean_agent_success,
                self.duplicate_pressure,
            ],
            dtype=float,
        )

    def discretize(self, bins: int = 4) -> tuple[int, ...]:
        """Bucketed key for tabular value functions."""
        raw = self.features()[1:]
        idx = np.clip((raw * bins).astype(int), 0, bins - 1)
        return tuple(int(v) for v in idx)

    # ------------------------------------------------------------- lifecycle
    def copy(self) -> "OrchestratorState":
        return replace(
            self,
            belief=list(self.belief),
            agent_history={k: AgentHistory.from_dict(v.to_dict()) for k, v in self.agent_history.items()},
            last_agents=list(self.last_agents),
            invocation_signatures=list(self.invocation_signatures),
        )

    def to_dict(self) -> dict:
        probabilities = self.belief_probabilities
        return {
            "step": self.step,
            "task_complexity": self.task_complexity,
            "belief": [float(v) for v in self.belief],
            "belief_probabilities": [float(v) for v in probabilities],
            "entropy": self.entropy,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "budget_total_usd": self.budget_total_usd,
            "budget_spent_usd": self.budget_spent_usd,
            "budget_remaining_usd": self.budget_remaining_usd,
            "budget_remaining": self.budget_remaining,
            "latency_budget_ms": self.latency_budget_ms,
            "latency_consumed_ms": self.latency_consumed_ms,
            "latency_remaining": self.latency_remaining,
            "tokens_consumed": self.tokens_consumed,
            "quality": self.quality,
            "verification_score": self.verification_score,
            "memory_coverage": self.memory_coverage,
            "total_subtasks": self.total_subtasks,
            "unresolved_subtasks": self.unresolved_subtasks,
            "unresolved_ratio": self.unresolved_ratio,
            "duplicate_pressure": self.duplicate_pressure,
            "mean_agent_success": self.mean_agent_success,
            "agent_history": {k: v.to_dict() for k, v in self.agent_history.items()},
            "last_agents": list(self.last_agents),
            "invocation_signatures": list(self.invocation_signatures),
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "latent_hypothesis": self.latent_hypothesis,
            "features": {
                name: float(value) for name, value in zip(FEATURE_NAMES, self.features(), strict=True)
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OrchestratorState":
        return cls(
            step=int(payload.get("step", 0)),
            task_complexity=float(payload.get("task_complexity", 0.5)),
            belief=[float(v) for v in payload.get("belief", [])],
            budget_total_usd=float(payload.get("budget_total_usd", 1.0)),
            budget_spent_usd=float(payload.get("budget_spent_usd", 0.0)),
            latency_budget_ms=float(payload.get("latency_budget_ms", 60_000.0)),
            latency_consumed_ms=float(payload.get("latency_consumed_ms", 0.0)),
            tokens_consumed=int(payload.get("tokens_consumed", 0)),
            quality=float(payload.get("quality", 0.05)),
            verification_score=float(payload.get("verification_score", 0.0)),
            memory_coverage=float(payload.get("memory_coverage", 0.05)),
            total_subtasks=int(payload.get("total_subtasks", 5)),
            unresolved_subtasks=int(payload.get("unresolved_subtasks", 5)),
            duplicate_pressure=float(payload.get("duplicate_pressure", 0.0)),
            agent_history={
                k: AgentHistory.from_dict(v) for k, v in payload.get("agent_history", {}).items()
            },
            last_agents=list(payload.get("last_agents", [])),
            invocation_signatures=list(payload.get("invocation_signatures", [])),
            terminated=bool(payload.get("terminated", False)),
            termination_reason=payload.get("termination_reason"),
            latent_hypothesis=int(payload.get("latent_hypothesis", 0)),
        )


def initial_state(
    *,
    task_complexity: float,
    budget_usd: float,
    latency_budget_ms: float,
    belief_dim: int,
    rng: np.random.Generator,
) -> OrchestratorState:
    """Sample S_0. The latent hypothesis is the unobserved ground truth the agents chase."""
    complexity = float(np.clip(task_complexity, 0.05, 0.99))
    subtasks = int(np.clip(round(3 + complexity * 9), 3, 14))
    # A weakly informative prior: uniform Dirichlet, so H(S_0) starts at the maximum.
    belief = list(np.full(belief_dim, 1.0, dtype=float))
    # Competence priors are agent-specific and get harder as task complexity rises.
    history = {
        spec.id: AgentHistory(
            alpha=spec.prior_alpha,
            beta=spec.prior_beta + complexity * 2.0,
        )
        for spec in AGENT_SPECS
    }
    return OrchestratorState(
        step=0,
        task_complexity=complexity,
        belief=belief,
        budget_total_usd=budget_usd,
        budget_spent_usd=0.0,
        latency_budget_ms=latency_budget_ms,
        latency_consumed_ms=0.0,
        tokens_consumed=0,
        quality=float(np.clip(0.08 - 0.05 * complexity, 0.0, 1.0)),
        verification_score=0.0,
        memory_coverage=float(rng.uniform(0.02, 0.12)),
        total_subtasks=subtasks,
        unresolved_subtasks=subtasks,
        duplicate_pressure=0.0,
        agent_history=history,
        last_agents=[],
        invocation_signatures=[],
        terminated=False,
        termination_reason=None,
        latent_hypothesis=int(rng.integers(0, belief_dim)),
    )
