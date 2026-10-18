"""
Comprehensive unit and regression tests for QuietFray v0.1.
Verifies all required scenarios:
- Initial state and empty event streams
- Fast and slow decay dynamics
- Chronologically ordered event sequences
- Repeated, rapid, and out-of-order events
- Confidence, diversity, and burst-similarity mark handling
- Downvotes, reopenings, and persistence behavior
- Finite outputs, numerical stability, and invalid inputs
- Subcriticality constraints
- Chronic-need, sudden-burst, and old-popular-issue ranking scenarios
"""

import math
import pytest
import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quietfray import QuietFrayConfig, QuietFrayEngine, QuietFrayState


class TestQuietFrayCore:

    def test_initial_state_and_empty_stream(self):
        """Initial state should produce a valid neutral score reflecting the category prior."""
        engine = QuietFrayEngine()
        state = QuietFrayState()

        assert state.last_t_days == 0.0
        assert state.upvotes == 0
        assert state.downvotes == 0
        assert state.reopens == 0
        assert state.vote_x_fast == 0.0
        assert state.vote_x_slow == 0.0

        # With default category prior = 0.3
        score = engine.compute_unified_score(state, current_age_days=0.0, category_prior=0.3)
        expected_logit = math.log(0.3 / 0.7)
        assert pytest.approx(score["unified_log_odds"], rel=1e-5) == expected_logit
        assert pytest.approx(score["posterior_prob_h1"], rel=1e-5) == 0.3

        # With 50% prior
        score_neutral = engine.compute_unified_score(state, current_age_days=0.0, category_prior=0.5)
        assert pytest.approx(score_neutral["unified_log_odds"], abs=1e-6) == 0.0
        assert pytest.approx(score_neutral["posterior_prob_h1"], abs=1e-6) == 0.5

    def test_fast_and_slow_decay(self):
        """Fast kernel (tau_f = 0.02 d) should decay rapidly; slow kernel (tau_s = 3.0 d) should persist."""
        engine = QuietFrayEngine()
        state = QuietFrayState()

        # Ingest one vote at t = 0
        engine.ingest_vote_event(state, t_days=0.0, is_upvote=True)
        assert state.vote_x_fast == 1.0
        assert state.vote_x_slow == 1.0

        # Advance by 0.1 days (2.4 hours: ~5 fast half-lives, but small fraction of slow half-life)
        engine.advance_time(state, t_days=0.1)

        # Fast decay: exp(-0.1 / 0.02) = exp(-5) ~ 0.006738
        assert state.vote_x_fast < 0.01
        # Slow decay: exp(-0.1 / 3.0) ~ 0.9672
        assert state.vote_x_slow > 0.95

    def test_chronologically_ordered_event_sequences(self):
        """Events arriving in chronological order should smoothly accumulate evidence."""
        engine = QuietFrayEngine()
        state = QuietFrayState()

        timestamps = [0.1, 0.5, 1.0, 2.0, 3.5]
        for t in timestamps:
            engine.ingest_report_event(state, t_days=t, confidence=0.9, diversity=0.8, burst_similarity=0.1)
            assert state.last_t_days == t

        score = engine.compute_unified_score(state, current_age_days=4.0)
        assert score["unified_log_odds"] > 0.0
        assert score["posterior_prob_h1"] > 0.5

    def test_repeated_and_rapid_events(self):
        """Microsecond-scale inter-arrivals must not trigger catastrophic cancellation or NaN."""
        engine = QuietFrayEngine()
        state = QuietFrayState()

        t_base = 1.0
        # 200 events spaced by 1 microsecond (1.157e-11 days)
        for i in range(200):
            t = t_base + i * 1e-9
            engine.ingest_vote_event(state, t_days=t, is_upvote=True)

        score = engine.compute_unified_score(state, current_age_days=1.0)
        assert not math.isnan(score["unified_log_odds"])
        assert not math.isinf(score["unified_log_odds"])
        assert 0.0 <= score["posterior_prob_h1"] <= 1.0
        assert state.upvotes == 200

    def test_out_of_order_events_handling(self):
        """Out-of-order events (t < last_t_days) should be handled gracefully without negative dt."""
        engine = QuietFrayEngine()
        state = QuietFrayState()

        engine.ingest_report_event(state, t_days=5.0, confidence=0.8, diversity=0.7, burst_similarity=0.1)
        assert state.last_t_days == 5.0

        # Arrives with timestamp 4.0 (< 5.0)
        engine.ingest_report_event(state, t_days=4.0, confidence=0.8, diversity=0.7, burst_similarity=0.1)
        assert state.last_t_days == 5.0  # last_t_days must not regress

        score = engine.compute_unified_score(state, current_age_days=5.0)
        assert not math.isnan(score["unified_log_odds"])

    def test_semantic_marks_handling(self):
        """
        High confidence & diversity with low burst similarity yields positive mark log-LR.
        Low confidence & diversity with high burst similarity yields negative mark log-LR.
        """
        engine = QuietFrayEngine()

        state_high_qual = QuietFrayState()
        q_high = engine.ingest_report_event(state_high_qual, t_days=1.0, confidence=0.95, diversity=0.90, burst_similarity=0.05)
        assert q_high > 0.7
        assert state_high_qual.report_mark_log_lr > 0.0

        state_low_qual = QuietFrayState()
        q_low = engine.ingest_report_event(state_low_qual, t_days=1.0, confidence=0.20, diversity=0.15, burst_similarity=0.95)
        assert q_low < 0.05
        assert state_low_qual.report_mark_log_lr < 0.0

        assert state_high_qual.report_mark_log_lr > state_low_qual.report_mark_log_lr

    def test_downvotes_and_reopen_penalties(self):
        """Downvotes decrease vote_dir_log_lr, while verified reopens increment reopen_log_lr."""
        engine = QuietFrayEngine()
        state = QuietFrayState()

        # Add 5 downvotes
        for i in range(5):
            engine.ingest_vote_event(state, t_days=float(i), is_upvote=False)

        assert state.downvotes == 5
        assert state.vote_dir_log_lr < 0.0

        # Add 2 reopen events (indicating solutions failed and issue reopened)
        engine.ingest_reopen_event(state, t_days=5.0)
        engine.ingest_reopen_event(state, t_days=6.0)

        assert state.reopens == 2
        expected_reopen_llr = 2 * math.log(0.70 / 0.20)
        assert pytest.approx(state.reopen_log_lr, abs=1e-6) == expected_reopen_llr

    def test_subcriticality_constraints(self):
        """Subcriticality constraint validation enforces nu < 1.0."""
        # Valid config
        valid_cfg = QuietFrayConfig()
        valid_cfg.validate_bounds()
        for ch, nu in valid_cfg.branching_ratios.items():
            assert nu < 1.0

        # Explosive config (nu >= 1.0) must raise ValueError
        with pytest.raises(ValueError, match="Subcriticality violation"):
            QuietFrayConfig(alpha_v_h0=(60.0, 0.025), tau_f=0.02)  # 60 * 0.02 = 1.2 >= 1.0

        with pytest.raises(ValueError, match="must be strictly positive"):
            QuietFrayConfig(tau_f=-0.01)

        with pytest.raises(ValueError, match="Fast timescale tau_f must be strictly less"):
            QuietFrayConfig(tau_f=4.0, tau_s=3.0)

    def test_invalid_and_nan_inputs(self):
        """NaN inputs must raise ValueError, out-of-range marks must be clamped safely."""
        engine = QuietFrayEngine()
        state = QuietFrayState()

        with pytest.raises(ValueError):
            engine.ingest_report_event(state, t_days=float("nan"), confidence=0.8, diversity=0.8, burst_similarity=0.1)

        with pytest.raises(ValueError):
            engine.ingest_vote_event(state, t_days=float("nan"))

        with pytest.raises(ValueError):
            engine.compute_unified_score(state, current_age_days=float("nan"))

        # Marks outside [0, 1] are clamped without error
        q = engine.ingest_report_event(state, t_days=1.0, confidence=1.5, diversity=-0.5, burst_similarity=2.0)
        assert 0.0 <= q <= 1.0

    def test_ranking_scenarios_chronic_vs_burst_vs_old_popular(self):
        """
        Ranking Test:
        - Issue 1 (Chronic Need): Steady reports over weeks, high diversity, moderate votes.
        - Issue 2 (Sudden Viral / Sybil Spike): Rapid burst in 1 hour, copy-paste similarity.
        - Issue 3 (Old Popular Legacy): Massive votes 180 days ago, zero activity since.
        QuietFray should rank: Chronic Need > Old Popular Legacy and Chronic Need > Viral Spike.
        """
        engine = QuietFrayEngine()

        # 1. Chronic Need
        state_chronic = QuietFrayState()
        for day in range(1, 21):
            engine.ingest_report_event(
                state_chronic,
                t_days=float(day),
                confidence=0.88,
                diversity=0.82,
                burst_similarity=0.15,
            )
            engine.ingest_vote_event(state_chronic, t_days=float(day) + 0.1, is_upvote=True)
        score_chronic = engine.compute_unified_score(state_chronic, current_age_days=21.0)

        # 2. Sudden Viral / Sybil Spike
        state_burst = QuietFrayState()
        t_burst = 20.0
        for i in range(20):
            t = t_burst + i * 0.001  # arrive within minutes
            engine.ingest_report_event(
                state_burst,
                t_days=t,
                confidence=0.55,
                diversity=0.20,
                burst_similarity=0.90,
            )
            engine.ingest_vote_event(state_burst, t_days=t + 0.0001, is_upvote=True)
        score_burst = engine.compute_unified_score(state_burst, current_age_days=21.0)

        # 3. Old Popular Legacy
        state_old = QuietFrayState()
        for i in range(50):
            engine.ingest_vote_event(state_old, t_days=0.01 * i, is_upvote=True)
        # Dormant for 180 days
        score_old = engine.compute_unified_score(state_old, current_age_days=180.0)

        # Verify ranking
        assert score_chronic["unified_log_odds"] > score_burst["unified_log_odds"]
        assert score_chronic["posterior_prob_h1"] > score_burst["posterior_prob_h1"]
        assert score_chronic["unified_log_odds"] > score_old["unified_log_odds"]
