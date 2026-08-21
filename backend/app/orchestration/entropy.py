"""Explicit information-theoretic bookkeeping.

The orchestrator maintains a Dirichlet belief over ``K`` competing solution hypotheses.
Entropy is computed on the posterior mean of that Dirichlet, in bits. Information gain for a
step is ``H(before) - H(after)`` and is reported to the UI alongside both endpoints so the
arithmetic is auditable.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def normalize(alpha: np.ndarray) -> np.ndarray:
    """Posterior mean of a Dirichlet with concentration ``alpha``."""
    alpha = np.asarray(alpha, dtype=float)
    total = float(alpha.sum())
    if total <= _EPS:
        return np.full(alpha.shape, 1.0 / max(alpha.size, 1))
    return alpha / total


def shannon_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = np.clip(np.asarray(probabilities, dtype=float), _EPS, 1.0)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def max_entropy(dim: int) -> float:
    return float(np.log2(max(dim, 2)))


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Entropy scaled to [0, 1]; used directly as the state's ``uncertainty`` field."""
    p = np.asarray(probabilities, dtype=float)
    return float(np.clip(shannon_entropy(p) / max_entropy(p.size), 0.0, 1.0))


def belief_entropy(alpha: np.ndarray) -> float:
    return shannon_entropy(normalize(alpha))


def information_gain(entropy_before: float, entropy_after: float) -> float:
    """H(S_t) - H(S_{t+1}) in bits. Negative values mean the step revealed new ambiguity."""
    return float(entropy_before - entropy_after)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(p || q) in bits; used as a secondary 'belief movement' diagnostic."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0)
    q = np.clip(np.asarray(q, dtype=float), _EPS, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log2(p / q)).sum())


def confidence_from_belief(alpha: np.ndarray) -> float:
    """Confidence = posterior mass on the leading hypothesis, tempered by evidence volume.

    A peaked-but-thin posterior (few observations) should not read as high confidence, so the
    leading mass is shrunk toward the uniform prior by the total observed concentration.
    """
    p = normalize(alpha)
    leader = float(p.max())
    uniform = 1.0 / p.size
    observations = max(float(np.asarray(alpha, dtype=float).sum()) - p.size, 0.0)
    evidence_weight = observations / (observations + 4.0)
    return float(np.clip(uniform + (leader - uniform) * evidence_weight, 0.0, 1.0))
