"""
Unit tests for classical ranking baseline implementations.
"""

import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quietfray.baselines import (
    raw_net_votes,
    wilson_lower_bound,
    exponential_recency_decay,
    empirical_bayes_vote_score,
)


def test_raw_net_votes():
    assert raw_net_votes(100, 20) == 80.0
    assert raw_net_votes(0, 5) == -5.0


def test_wilson_lower_bound():
    # Empty
    assert wilson_lower_bound(0, 0) == 0.0
    # 10 upvotes, 0 downvotes
    w = wilson_lower_bound(10, 0)
    assert 0.0 < w < 1.0
    # Perfect large sample approaches 1.0
    w_large = wilson_lower_bound(1000, 0)
    assert w_large > 0.99


def test_exponential_recency_decay():
    events = [(0.0, 1), (1.0, 1), (2.0, -1)]
    score = exponential_recency_decay(events, current_time_days=2.0, tau_days=3.0)
    assert not math.isnan(score)


def test_empirical_bayes_vote_score():
    score = empirical_bayes_vote_score(10, 2, alpha=2.0, beta=2.0)
    expected = math.log((10 + 2) / (2 + 2))  # log(12/4) = log(3)
    assert math.isclose(score, expected, rel_tol=1e-5)
