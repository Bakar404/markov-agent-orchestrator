"""Policy registry — the four evolution stages plus baselines."""

from __future__ import annotations

from .bandit import LinUCBPolicy
from .base import Policy, masked_softmax
from .baselines import HeuristicPolicy, RandomPolicy
from .external import ExternalPolicy
from .fixed_sequence import FixedSequencePolicy
from .markov_game import CooperativeMarkovGamePolicy
from .marl import MultiAgentRLPolicy
from .mdp import MDPQLearningPolicy
from .single_agent import SingleAgentPolicy

POLICY_REGISTRY: dict[str, type[Policy]] = {
    RandomPolicy.id: RandomPolicy,
    SingleAgentPolicy.id: SingleAgentPolicy,
    ExternalPolicy.id: ExternalPolicy,
    HeuristicPolicy.id: HeuristicPolicy,
    FixedSequencePolicy.id: FixedSequencePolicy,
    LinUCBPolicy.id: LinUCBPolicy,
    MDPQLearningPolicy.id: MDPQLearningPolicy,
    CooperativeMarkovGamePolicy.id: CooperativeMarkovGamePolicy,
    MultiAgentRLPolicy.id: MultiAgentRLPolicy,
}

DEFAULT_POLICY = LinUCBPolicy.id


def create_policy(policy_id: str, *, feature_dim: int, **kwargs: object) -> Policy:
    try:
        cls = POLICY_REGISTRY[policy_id]
    except KeyError as exc:
        known = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"Unknown policy '{policy_id}'. Available: {known}") from exc
    return cls(feature_dim=feature_dim, **kwargs)


def policy_catalog() -> list[dict]:
    catalog = [
        {
            "id": cls.id,
            "label": cls.label,
            "stage": cls.stage,
            "family": cls.family,
            "description": cls.description,
        }
        for cls in POLICY_REGISTRY.values()
    ]
    catalog.sort(key=lambda item: (item["stage"], item["id"]))
    return catalog


__all__ = [
    "POLICY_REGISTRY",
    "DEFAULT_POLICY",
    "Policy",
    "create_policy",
    "policy_catalog",
    "masked_softmax",
    "RandomPolicy",
    "SingleAgentPolicy",
    "ExternalPolicy",
    "HeuristicPolicy",
    "FixedSequencePolicy",
    "LinUCBPolicy",
    "MDPQLearningPolicy",
    "CooperativeMarkovGamePolicy",
    "MultiAgentRLPolicy",
]
