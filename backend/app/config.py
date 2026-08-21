"""Application configuration.

Settings are plain environment variables so the service stays dependency-light and
container-friendly. A ``.env`` file next to ``backend/`` is loaded when present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESEARCH_DIR = ROOT_DIR / "research"

load_dotenv(BACKEND_DIR / ".env")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RewardWeights:
    """Weights for each additive term of the reward function."""

    quality: float = 1.4
    verification: float = 1.1
    information_gain: float = 1.6
    cost: float = 1.0
    latency: float = 0.6
    duplicate: float = 0.9
    progress: float = 0.8
    terminal: float = 2.5

    @classmethod
    def from_env(cls) -> "RewardWeights":
        return cls(
            quality=_env_float("REWARD_W_QUALITY", 1.4),
            verification=_env_float("REWARD_W_VERIFICATION", 1.1),
            information_gain=_env_float("REWARD_W_INFO_GAIN", 1.6),
            cost=_env_float("REWARD_W_COST", 1.0),
            latency=_env_float("REWARD_W_LATENCY", 0.6),
            duplicate=_env_float("REWARD_W_DUPLICATE", 0.9),
            progress=_env_float("REWARD_W_PROGRESS", 0.8),
            terminal=_env_float("REWARD_W_TERMINAL", 2.5),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "verification": self.verification,
            "information_gain": self.information_gain,
            "cost": self.cost,
            "latency": self.latency,
            "duplicate": self.duplicate,
            "progress": self.progress,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class Settings:
    app_name: str = "Markov Agent Orchestrator"
    version: str = "0.1.0"
    database_url: str = ""
    cors_origins: tuple[str, ...] = ()
    default_seed: int = 20260820
    max_steps: int = 60
    belief_dim: int = 8
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    research_allow_network: bool = True
    research_http_timeout: float = 12.0
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_api_key: str = ""
    papers_with_code_base_url: str = "https://paperswithcode.com/api/v1"
    hits_mcp_endpoint: str = ""
    hits_mcp_api_key: str = ""
    hits_mcp_tool: str = "research.search"

    @property
    def seed_corpus_path(self) -> Path:
        return RESEARCH_DIR / "seed_corpus.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    default_db = f"sqlite:///{(DATA_DIR / 'orchestrator.db').as_posix()}"
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return Settings(
        database_url=os.getenv("DATABASE_URL", default_db),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        default_seed=_env_int("DEFAULT_SEED", 20260820),
        max_steps=_env_int("MAX_STEPS", 60),
        belief_dim=max(3, _env_int("BELIEF_DIM", 8)),
        reward_weights=RewardWeights.from_env(),
        research_allow_network=_env_bool("RESEARCH_ALLOW_NETWORK", True),
        research_http_timeout=_env_float("RESEARCH_HTTP_TIMEOUT", 12.0),
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
        hits_mcp_endpoint=os.getenv("HITS_MCP_ENDPOINT", ""),
        hits_mcp_api_key=os.getenv("HITS_MCP_API_KEY", ""),
        hits_mcp_tool=os.getenv("HITS_MCP_TOOL", "research.search"),
    )
