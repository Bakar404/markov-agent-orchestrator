"""Research taxonomy: categories, keyword signatures, and standing discovery queries."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    name: str
    description: str
    color: str
    keywords: tuple[str, ...]
    discovery_queries: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        name="Markov Games",
        description="Games where state transitions follow a Markov process over joint actions.",
        color="#38bdf8",
        keywords=(
            "markov game",
            "markov games",
            "general-sum game",
            "nash q",
            "minimax-q",
            "friend-or-foe",
            "equilibrium",
        ),
        discovery_queries=(
            "Markov games multi-agent reinforcement learning",
            "Nash equilibrium learning in general-sum Markov games",
        ),
    ),
    Category(
        name="Stochastic Games",
        description="Shapley's stochastic games and their modern learning-theoretic treatments.",
        color="#818cf8",
        keywords=("stochastic game", "stochastic games", "shapley", "discounted game"),
        discovery_queries=(
            "stochastic games learning equilibria",
            "sample complexity of stochastic games",
        ),
    ),
    Category(
        name="MARL",
        description="Multi-agent reinforcement learning algorithms and value factorization.",
        color="#a78bfa",
        keywords=(
            "multi-agent reinforcement learning",
            "marl",
            "qmix",
            "vdn",
            "value decomposition",
            "maddpg",
            "coma",
            "mappo",
            "centralized training",
            "credit assignment",
        ),
        discovery_queries=(
            "multi-agent reinforcement learning value decomposition",
            "cooperative multi-agent reinforcement learning credit assignment",
        ),
    ),
    Category(
        name="Agent Orchestration",
        description="Frameworks that coordinate multiple LLM agents toward a shared objective.",
        color="#f472b6",
        keywords=(
            "multi-agent conversation",
            "agent framework",
            "orchestration",
            "agent collaboration",
            "role-playing agents",
            "agent debate",
            "agent society",
            "swarm",
        ),
        discovery_queries=(
            "LLM multi-agent orchestration framework",
            "optimizing multi-agent LLM systems as graphs",
        ),
    ),
    Category(
        name="Reinforcement Learning",
        description="Core RL: value functions, policy gradients, exploration and reward shaping.",
        color="#34d399",
        keywords=(
            "reinforcement learning",
            "q-learning",
            "policy gradient",
            "temporal difference",
            "reward shaping",
            "bandit",
            "exploration",
            "markov decision process",
        ),
        discovery_queries=(
            "potential-based reward shaping theory",
            "contextual bandits linear payoff regret",
        ),
    ),
    Category(
        name="Planning",
        description="Search, tree-based planning and hierarchical decomposition.",
        color="#fbbf24",
        keywords=(
            "monte carlo tree search",
            "mcts",
            "uct",
            "planning",
            "tree of thoughts",
            "options framework",
            "temporal abstraction",
            "pomdp",
        ),
        discovery_queries=(
            "Monte Carlo tree search planning survey",
            "hierarchical planning temporal abstraction options",
        ),
    ),
    Category(
        name="Multi-Agent Systems",
        description="Classical MAS: coordination, communication protocols and social choice.",
        color="#22d3ee",
        keywords=(
            "multiagent system",
            "multi-agent system",
            "coordination",
            "decentralized",
            "dec-pomdp",
            "cooperative agents",
        ),
        discovery_queries=(
            "decentralized POMDP coordination",
            "multiagent systems survey machine learning perspective",
        ),
    ),
    Category(
        name="Tool Use",
        description="Agents that call external tools, APIs and retrieval systems.",
        color="#fb923c",
        keywords=(
            "tool use",
            "tool learning",
            "api call",
            "function calling",
            "retrieval augmented",
            "toolformer",
            "react",
        ),
        discovery_queries=(
            "language model tool use API calling",
            "retrieval augmented generation for agents",
        ),
    ),
    Category(
        name="Agent Routing",
        description="Selecting which model or agent handles a request under cost constraints.",
        color="#f87171",
        keywords=(
            "routing",
            "router",
            "model selection",
            "cascade",
            "cost-efficient inference",
            "frugal",
            "mixture of experts",
        ),
        discovery_queries=(
            "LLM routing cost quality tradeoff",
            "cascaded model selection for language models",
        ),
    ),
)

CATEGORY_BY_NAME: dict[str, Category] = {c.name: c for c in CATEGORIES}
CATEGORY_NAMES: tuple[str, ...] = tuple(c.name for c in CATEGORIES)

#: Baseline topic of this project. Seeded papers are scored against it so the library has a
#: meaningful ranking before the user has searched for anything.
PROJECT_QUERY = (
    "markov game stochastic game multi-agent reinforcement learning contextual bandit "
    "agent orchestration routing planning reward shaping information gain"
)


def taxonomy_catalog() -> list[dict]:
    return [
        {
            "name": c.name,
            "description": c.description,
            "color": c.color,
            "keywords": list(c.keywords),
            "discovery_queries": list(c.discovery_queries),
        }
        for c in CATEGORIES
    ]


def classify(title: str, abstract: str = "", existing: list[str] | None = None) -> list[str]:
    """Assign taxonomy categories by keyword signature over title + abstract."""
    haystack = f"{title} {abstract}".lower()
    matched = {tag for tag in (existing or []) if tag in CATEGORY_BY_NAME}
    for category in CATEGORIES:
        if any(keyword in haystack for keyword in category.keywords):
            matched.add(category.name)
    if not matched:
        matched.add("Reinforcement Learning" if "learning" in haystack else "Multi-Agent Systems")
    return sorted(matched)


_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def relevance_score(
    *,
    title: str,
    abstract: str,
    query: str,
    year: int | None,
    citation_count: int,
    tags: list[str],
    current_year: int = 2026,
) -> float:
    """Blend query overlap, taxonomy coverage, citation weight and recency into [0, 1].

    The components are deliberately transparent so the Research Library can explain a ranking
    instead of presenting an opaque score.
    """
    query_tokens = tokenize(query)
    doc_tokens = tokenize(f"{title} {abstract}")
    overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0

    taxonomy_coverage = min(len(tags) / 3.0, 1.0)

    # log-scaled so a 40k-citation classic does not swamp everything else
    citation_weight = min((citation_count**0.5) / 60.0, 1.0)

    age = max(current_year - (year or current_year), 0)
    recency = 1.0 / (1.0 + age / 8.0)

    score = 0.42 * overlap + 0.22 * taxonomy_coverage + 0.22 * citation_weight + 0.14 * recency
    return round(min(max(score, 0.0), 1.0), 4)
