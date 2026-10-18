"""
Regression test validating QuietFrayEngine against `fixtures/quietfray_test_vectors_v0.1.json`.
Ensures cross-language parity for TypeScript (MendSeed/MendSeed) and Python implementations.
"""

import json
import math
import os
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quietfray import QuietFrayConfig, QuietFrayEngine, QuietFrayState


def test_cross_language_test_vectors():
    fixture_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "fixtures", "quietfray_test_vectors_v0.1.json")
    )
    assert os.path.exists(fixture_path), f"Fixture not found at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data["config"]
    config = QuietFrayConfig(
        tau_f=cfg["tau_f"],
        tau_s=cfg["tau_s"],
        alpha_v_h0=tuple(cfg["alpha_v_h0"]),
        alpha_v_h1=tuple(cfg["alpha_v_h1"]),
        alpha_r_h0=tuple(cfg["alpha_r_h0"]),
        alpha_r_h1=tuple(cfg["alpha_r_h1"]),
        mu_vote=cfg["mu_vote"],
        mu_report=cfg["mu_report"],
        vote_a0=cfg["vote_a0"],
        vote_b0=cfg["vote_b0"],
        vote_a1=cfg["vote_a1"],
        vote_b1=cfg["vote_b1"],
        weibull_h0=tuple(cfg["weibull_h0"]),
        weibull_h1=tuple(cfg["weibull_h1"]),
        reopen_p1=cfg["reopen_p1"],
        reopen_p0=cfg["reopen_p0"],
    )
    engine = QuietFrayEngine(config=config)

    for case in data["cases"]:
        state = QuietFrayState()
        for ev in case["events"]:
            ev_type = ev["type"]
            if ev_type == "report":
                engine.ingest_report_event(
                    state,
                    t_days=ev["t_days"],
                    confidence=ev["confidence"],
                    diversity=ev["diversity"],
                    burst_similarity=ev["burst_similarity"],
                )
            elif ev_type == "vote":
                engine.ingest_vote_event(
                    state,
                    t_days=ev["t_days"],
                    is_upvote=ev["is_upvote"],
                )
            elif ev_type == "reopen":
                engine.ingest_reopen_event(
                    state,
                    t_days=ev["t_days"],
                )

        # Check final state
        expected_st = case["expected_final_state"]
        assert pytest.approx(state.last_t_days, abs=1e-5) == expected_st["last_t_days"]
        assert pytest.approx(state.vote_x_fast, abs=1e-5) == expected_st["vote_x_fast"]
        assert pytest.approx(state.vote_x_slow, abs=1e-5) == expected_st["vote_x_slow"]
        assert pytest.approx(state.report_x_fast, abs=1e-5) == expected_st["report_x_fast"]
        assert pytest.approx(state.report_x_slow, abs=1e-5) == expected_st["report_x_slow"]
        assert pytest.approx(state.vote_log_lr, abs=1e-5) == expected_st["vote_log_lr"]
        assert pytest.approx(state.report_log_lr, abs=1e-5) == expected_st["report_log_lr"]
        assert pytest.approx(state.report_mark_log_lr, abs=1e-5) == expected_st["report_mark_log_lr"]
        assert pytest.approx(state.vote_dir_log_lr, abs=1e-5) == expected_st["vote_dir_log_lr"]
        assert pytest.approx(state.reopen_log_lr, abs=1e-5) == expected_st["reopen_log_lr"]
        assert state.upvotes == expected_st["upvotes"]
        assert state.downvotes == expected_st["downvotes"]
        assert state.reopens == expected_st["reopens"]

        # Check evaluations
        for eval_case in case["evaluations"]:
            res = engine.compute_unified_score(
                state,
                current_age_days=eval_case["current_age_days"],
                category_prior=eval_case["category_prior"],
                current_horizon_days=eval_case.get("current_horizon_days"),
                has_reopen=eval_case.get("has_reopen"),
            )
            assert pytest.approx(res["unified_log_odds"], abs=1e-5) == eval_case["unified_log_odds"]
            assert pytest.approx(res["posterior_prob_h1"], abs=1e-5) == eval_case["posterior_prob_h1"]
