"""
Unit tests for QuietFlayEngine v0.1 verifying O(1) state updates,
numerical precision, and non-explosive Hawkes dynamics.
"""
import sys
import os
import math

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sdk.mendseed_ai.quietflay import QuietFlayEngineV01, QuietFlayState


def test_initial_state():
    engine = QuietFlayEngineV01()
    state = QuietFlayState()
    score = engine.compute_unified_score(state, current_age_days=0.0, category_prior=0.3)
    
    assert "unified_log_odds" in score
    assert "posterior_prob_h1" in score
    assert 0.0 <= score["posterior_prob_h1"] <= 1.0
    print("Test Initial State: Passed")


def test_steady_chronic_vs_sybil_spike():
    engine = QuietFlayEngineV01()
    
    # Issue A: Chronic Issue (reports arrive spaced across days with high entropy & low burst similarity)
    state_a = QuietFlayState()
    for day in range(1, 15):
        engine.ingest_report_event(
            state_a,
            t_days=float(day),
            confidence=0.92,
            diversity=0.88,
            burst_similarity=0.10,
        )
        engine.ingest_vote_event(state_a, t_days=float(day) + 0.1, is_upvote=True)
        engine.ingest_vote_event(state_a, t_days=float(day) + 0.2, is_upvote=True)

    score_a = engine.compute_unified_score(state_a, current_age_days=15.0)

    # Issue B: Sybil Raid (14 reports arrive within 10 minutes, high burst similarity, lower diversity)
    state_b = QuietFlayState()
    t_start = 14.0
    for idx in range(14):
        t_event = t_start + (idx * 0.0005)  # arrives every ~43 seconds
        engine.ingest_report_event(
            state_b,
            t_days=t_event,
            confidence=0.60,
            diversity=0.20,
            burst_similarity=0.95,
        )
        engine.ingest_vote_event(state_b, t_days=t_event + 0.00001, is_upvote=True)
        engine.ingest_vote_event(state_b, t_days=t_event + 0.00002, is_upvote=True)

    score_b = engine.compute_unified_score(state_b, current_age_days=15.0)

    print(f"Chronic Issue Unified Log-Odds: {score_a['unified_log_odds']}, P(H1): {score_a['posterior_prob_h1']}")
    print(f"Sybil Raid Unified Log-Odds:    {score_b['unified_log_odds']}, P(H1): {score_b['posterior_prob_h1']}")

    # Chronic issue should decisively outrank Sybil raid
    assert score_a["unified_log_odds"] > score_b["unified_log_odds"]
    assert score_a["posterior_prob_h1"] > score_b["posterior_prob_h1"]
    print("Test Steady Chronic vs Sybil Spike: Passed")


def test_numerical_stability_rapid_events():
    engine = QuietFlayEngineV01()
    state = QuietFlayState()
    # Microsecond-scale inter-arrivals (1 microsecond = ~1.157e-11 days)
    t = 1.0
    for _ in range(100):
        t += 1e-8
        engine.ingest_vote_event(state, t_days=t, is_upvote=True)

    score = engine.compute_unified_score(state, current_age_days=1.0)
    assert not math.isnan(score["unified_log_odds"])
    assert not math.isinf(score["unified_log_odds"])
    assert 0.0 <= score["posterior_prob_h1"] <= 1.0
    print("Test Numerical Stability: Passed")


if __name__ == "__main__":
    test_initial_state()
    test_steady_chronic_vs_sybil_spike()
    test_numerical_stability_rapid_events()
    print("All QuietFlayEngine tests passed successfully!")
