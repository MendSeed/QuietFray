"""
Classical issue ranking baselines implemented for empirical benchmark comparison.
Reference: QuietFray v0.1 Specification, Section 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


def raw_net_votes(upvotes: int, downvotes: int) -> float:
    """Baseline 1: S_raw = U - D"""
    return float(upvotes - downvotes)


def wilson_lower_bound(upvotes: int, downvotes: int, confidence: float = 0.95) -> float:
    """
    Baseline 2: Wilson Lower Confidence Bound on Bernoulli proportion.
    Default 95% confidence (z = 1.95996).
    """
    n = upvotes + downvotes
    if n == 0:
        return 0.0
    p_hat = upvotes / n
    z = 1.959963984540054  # Normal quantile for 95%
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return (center - spread) / denom


def exponential_recency_decay(
    vote_events: List[Tuple[float, int]],  # List of (timestamp_days, direction: +1 or -1)
    current_time_days: float,
    tau_days: float = 3.0,
) -> float:
    """
    Baseline 3: Single-kernel exponential recency decay.
    S_decay = sum_{j: t_j <= T} d_j * exp(-(T - t_j) / tau)
    """
    score = 0.0
    for t_j, direction in vote_events:
        if t_j <= current_time_days:
            dt = current_time_days - t_j
            score += float(direction) * math.exp(-dt / tau_days)
    return score


def empirical_bayes_vote_score(
    upvotes: int,
    downvotes: int,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> float:
    """
    Baseline 4: Beta-smoothed log-odds score.
    S_EB = log((U + alpha) / (D + beta))
    """
    u_smoothed = upvotes + alpha
    d_smoothed = downvotes + beta
    return math.log(u_smoothed / d_smoothed)
