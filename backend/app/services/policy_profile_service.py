"""Policy profiles: the router's memory across episodes.

Each run builds a fresh policy, so by default nothing a policy learns survives the episode that
taught it. A profile lifts those parameters out: train one in simulation with
``tools/campaign.py``, then load it to route real work.

Loading is deliberately guarded. LinUCB stores a ``feature_dim x feature_dim`` ridge matrix per
action, so weights fitted against a different context vector are not merely stale, they are the
wrong shape. Every profile records the signature it was fitted for and refuses to load into a
policy that does not match.

The campaign measures that carrying parameters is worth +1.13 for MARL but **-0.80** for the
contextual bandit, so profiles are per-policy and resettable rather than a single shared blob.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import PolicyProfile
from ..orchestration.actions import ACTIONS
from ..orchestration.engine import OrchestrationEngine
from ..orchestration.state import FEATURE_DIM

DEFAULT_PROFILE = "default"


class ProfileSignatureMismatch(ValueError):
    pass


def signature_for(policy_id: str, feature_dim: int = FEATURE_DIM) -> str:
    return f"{policy_id}:d{feature_dim}:a{len(ACTIONS)}"


class PolicyProfileService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, name: str, policy_id: str) -> PolicyProfile | None:
        return self.session.scalar(
            select(PolicyProfile).where(
                PolicyProfile.name == name, PolicyProfile.policy == policy_id
            )
        )

    def list_profiles(self) -> list[dict]:
        rows = self.session.scalars(
            select(PolicyProfile).order_by(PolicyProfile.name, PolicyProfile.policy)
        ).all()
        return [self._summary(row) for row in rows]

    def apply_to(self, engine: OrchestrationEngine, name: str) -> bool:
        """Load learned parameters into a freshly built engine. Returns False when absent."""
        profile = self.get(name, engine.config.policy)
        if profile is None:
            return False

        expected = signature_for(engine.config.policy)
        if profile.signature != expected:
            raise ProfileSignatureMismatch(
                f"profile '{name}' was fitted for {profile.signature}, "
                f"but this policy needs {expected}. Reset the profile to retrain it."
            )

        engine.policy.load_state_dict(profile.state)
        return True

    def capture(self, engine: OrchestrationEngine, name: str, *, episode_reward: float) -> dict:
        """Persist the policy's parameters after an episode and fold in its statistics."""
        policy_id = engine.config.policy
        profile = self.get(name, policy_id)
        signature = signature_for(policy_id)

        if profile is None:
            # Column defaults only land at INSERT, so seed the counters for the in-memory object.
            profile = PolicyProfile(
                name=name,
                policy=policy_id,
                signature=signature,
                state={},
                episodes=0,
                total_steps=0,
                cumulative_reward=0.0,
                mean_episode_reward=0.0,
                notes="",
            )
            self.session.add(profile)
        elif profile.signature != signature:
            raise ProfileSignatureMismatch(
                f"profile '{name}' holds {profile.signature}, refusing to overwrite with {signature}"
            )

        profile.state = engine.policy.state_dict()
        profile.episodes += 1
        profile.total_steps += engine.state.step
        profile.cumulative_reward += float(episode_reward)
        profile.mean_episode_reward = profile.cumulative_reward / max(profile.episodes, 1)
        self.session.commit()
        return self._summary(profile)

    def reset(self, name: str, policy_id: str) -> bool:
        profile = self.get(name, policy_id)
        if profile is None:
            return False
        self.session.delete(profile)
        self.session.commit()
        return True

    @staticmethod
    def _summary(row: PolicyProfile) -> dict:
        return {
            "id": row.id,
            "name": row.name,
            "policy": row.policy,
            "signature": row.signature,
            "episodes": row.episodes,
            "total_steps": row.total_steps,
            "cumulative_reward": row.cumulative_reward,
            "mean_episode_reward": row.mean_episode_reward,
            "notes": row.notes,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
