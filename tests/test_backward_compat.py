"""
Regression tests verifying full backward compatibility for existing consumers
importing from `sdk.mendseed_ai.quietflay`.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sdk.mendseed_ai.quietflay import QuietFlayEngineV01, QuietFlayState, QuietFlayEngine
from quietfray import QuietFrayEngine, QuietFrayState


def test_backward_compatibility_alias():
    """Ensure QuietFlayEngineV01 behaves identically to QuietFrayEngine."""
    assert QuietFlayEngine is QuietFrayEngine
    assert QuietFlayEngineV01 is QuietFrayEngine
    assert QuietFlayState is QuietFrayState

    legacy_engine = QuietFlayEngineV01()
    legacy_state = QuietFlayState()

    mark = legacy_engine.ingest_report_event(
        legacy_state,
        t_days=1.0,
        confidence=0.85,
        diversity=0.80,
        burst_similarity=0.15,
    )
    assert 0.0 <= mark <= 1.0

    legacy_engine.ingest_vote_event(legacy_state, t_days=1.1, is_upvote=True)
    score = legacy_engine.compute_unified_score(legacy_state, current_age_days=1.5)

    assert "unified_log_odds" in score
    assert "posterior_prob_h1" in score
    assert 0.0 <= score["posterior_prob_h1"] <= 1.0
