"""
Fixture generator for QuietFray v0.1 reproducible cross-language test vectors.
Generates `fixtures/quietfray_test_vectors_v0.1.json` for validation in
TypeScript (MendSeed/MendSeed) and Python (MendSeed/ai, MendSeed/QuietFray).

Command:
    py scripts/generate_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quietfray import QuietFrayConfig, QuietFrayEngine, QuietFrayState


def build_fixtures() -> dict:
    config = QuietFrayConfig()
    engine = QuietFrayEngine(config=config)

    test_vectors = {
        "version": "0.1.0",
        "model": "QuietFray v0.1",
        "description": "Standard cross-language reference test vectors for MendSeed platform implementations.",
        "config": {
            "tau_f": config.tau_f,
            "tau_s": config.tau_s,
            "alpha_v_h0": list(config.alpha_v_h0),
            "alpha_v_h1": list(config.alpha_v_h1),
            "alpha_r_h0": list(config.alpha_r_h0),
            "alpha_r_h1": list(config.alpha_r_h1),
            "mu_vote": config.mu_vote,
            "mu_report": config.mu_report,
            "vote_a0": config.vote_a0,
            "vote_b0": config.vote_b0,
            "vote_a1": config.vote_a1,
            "vote_b1": config.vote_b1,
            "weibull_h0": list(config.weibull_h0),
            "weibull_h1": list(config.weibull_h1),
            "reopen_p1": config.reopen_p1,
            "reopen_p0": config.reopen_p0,
        },
        "cases": [],
    }

    # Case 1: Empty / Cold-start
    s1 = QuietFrayState()
    score_1 = engine.compute_unified_score(s1, current_age_days=0.0, category_prior=0.3)
    test_vectors["cases"].append({
        "id": "cold_start_neutral",
        "description": "Evaluation of brand new issue with zero events under default 0.3 prior.",
        "events": [],
        "expected_final_state": {
            "last_t_days": s1.last_t_days,
            "vote_x_fast": s1.vote_x_fast,
            "vote_x_slow": s1.vote_x_slow,
            "report_x_fast": s1.report_x_fast,
            "report_x_slow": s1.report_x_slow,
            "vote_log_lr": s1.vote_log_lr,
            "report_log_lr": s1.report_log_lr,
            "report_mark_log_lr": s1.report_mark_log_lr,
            "vote_dir_log_lr": s1.vote_dir_log_lr,
            "reopen_log_lr": s1.reopen_log_lr,
            "upvotes": s1.upvotes,
            "downvotes": s1.downvotes,
            "reopens": s1.reopens,
        },
        "evaluations": [
            {
                "current_age_days": 0.0,
                "category_prior": 0.3,
                "unified_log_odds": score_1["unified_log_odds"],
                "posterior_prob_h1": score_1["posterior_prob_h1"],
            }
        ],
    })

    # Case 2: Single High Quality Report
    s2 = QuietFrayState()
    q2 = engine.ingest_report_event(s2, t_days=1.5, confidence=0.92, diversity=0.85, burst_similarity=0.10)
    score_2 = engine.compute_unified_score(s2, current_age_days=2.0, category_prior=0.3)
    test_vectors["cases"].append({
        "id": "single_report_high_quality",
        "description": "Single high-confidence diverse report with low burst similarity.",
        "events": [
            {"type": "report", "t_days": 1.5, "confidence": 0.92, "diversity": 0.85, "burst_similarity": 0.10, "mark_q": round(q2, 6)}
        ],
        "expected_final_state": {
            "last_t_days": s2.last_t_days,
            "vote_x_fast": s2.vote_x_fast,
            "vote_x_slow": s2.vote_x_slow,
            "report_x_fast": round(s2.report_x_fast, 8),
            "report_x_slow": round(s2.report_x_slow, 8),
            "vote_log_lr": round(s2.vote_log_lr, 8),
            "report_log_lr": round(s2.report_log_lr, 8),
            "report_mark_log_lr": round(s2.report_mark_log_lr, 8),
            "vote_dir_log_lr": round(s2.vote_dir_log_lr, 8),
            "reopen_log_lr": round(s2.reopen_log_lr, 8),
            "upvotes": s2.upvotes,
            "downvotes": s2.downvotes,
            "reopens": s2.reopens,
        },
        "evaluations": [
            {
                "current_age_days": 2.0,
                "category_prior": 0.3,
                "unified_log_odds": score_2["unified_log_odds"],
                "posterior_prob_h1": score_2["posterior_prob_h1"],
            }
        ],
    })

    # Case 3: Steady Chronic Need (Chronological multi-day arrivals)
    s3 = QuietFrayState()
    events_3 = []
    for day in range(1, 6):
        t_rep = float(day)
        q = engine.ingest_report_event(s3, t_days=t_rep, confidence=0.90, diversity=0.80, burst_similarity=0.15)
        events_3.append({"type": "report", "t_days": t_rep, "confidence": 0.90, "diversity": 0.80, "burst_similarity": 0.15, "mark_q": round(q, 6)})
        t_vote = t_rep + 0.2
        engine.ingest_vote_event(s3, t_days=t_vote, is_upvote=True)
        events_3.append({"type": "vote", "t_days": t_vote, "is_upvote": True})

    score_3 = engine.compute_unified_score(s3, current_age_days=6.0, category_prior=0.3)
    test_vectors["cases"].append({
        "id": "steady_chronic_need",
        "description": "5-day sustained sequence of reports and upvotes with steady persistence.",
        "events": events_3,
        "expected_final_state": {
            "last_t_days": s3.last_t_days,
            "vote_x_fast": round(s3.vote_x_fast, 8),
            "vote_x_slow": round(s3.vote_x_slow, 8),
            "report_x_fast": round(s3.report_x_fast, 8),
            "report_x_slow": round(s3.report_x_slow, 8),
            "vote_log_lr": round(s3.vote_log_lr, 8),
            "report_log_lr": round(s3.report_log_lr, 8),
            "report_mark_log_lr": round(s3.report_mark_log_lr, 8),
            "vote_dir_log_lr": round(s3.vote_dir_log_lr, 8),
            "reopen_log_lr": round(s3.reopen_log_lr, 8),
            "upvotes": s3.upvotes,
            "downvotes": s3.downvotes,
            "reopens": s3.reopens,
        },
        "evaluations": [
            {
                "current_age_days": 6.0,
                "category_prior": 0.3,
                "unified_log_odds": score_3["unified_log_odds"],
                "posterior_prob_h1": score_3["posterior_prob_h1"],
            }
        ],
    })

    # Case 4: Coordinated Sybil Raid (Rapid sub-hour burst, high repetition)
    s4 = QuietFrayState()
    events_4 = []
    t_start = 10.0
    for i in range(10):
        t_rep = t_start + i * 0.0005  # ~43s spacing
        q = engine.ingest_report_event(s4, t_days=t_rep, confidence=0.60, diversity=0.20, burst_similarity=0.95)
        events_4.append({"type": "report", "t_days": t_rep, "confidence": 0.60, "diversity": 0.20, "burst_similarity": 0.95, "mark_q": round(q, 6)})
        t_vote = t_rep + 0.00005
        engine.ingest_vote_event(s4, t_days=t_vote, is_upvote=True)
        events_4.append({"type": "vote", "t_days": t_vote, "is_upvote": True})

    score_4 = engine.compute_unified_score(s4, current_age_days=10.5, category_prior=0.3)
    test_vectors["cases"].append({
        "id": "sybil_burst_short_window",
        "description": "10 reports arriving within minutes with high burst similarity and low diversity.",
        "events": events_4,
        "expected_final_state": {
            "last_t_days": round(s4.last_t_days, 8),
            "vote_x_fast": round(s4.vote_x_fast, 8),
            "vote_x_slow": round(s4.vote_x_slow, 8),
            "report_x_fast": round(s4.report_x_fast, 8),
            "report_x_slow": round(s4.report_x_slow, 8),
            "vote_log_lr": round(s4.vote_log_lr, 8),
            "report_log_lr": round(s4.report_log_lr, 8),
            "report_mark_log_lr": round(s4.report_mark_log_lr, 8),
            "vote_dir_log_lr": round(s4.vote_dir_log_lr, 8),
            "reopen_log_lr": round(s4.reopen_log_lr, 8),
            "upvotes": s4.upvotes,
            "downvotes": s4.downvotes,
            "reopens": s4.reopens,
        },
        "evaluations": [
            {
                "current_age_days": 10.5,
                "category_prior": 0.3,
                "unified_log_odds": score_4["unified_log_odds"],
                "posterior_prob_h1": score_4["posterior_prob_h1"],
            }
        ],
    })

    # Case 5: Contested issue with downvotes and reopenings
    s5 = QuietFrayState()
    engine.ingest_vote_event(s5, t_days=1.0, is_upvote=True)
    engine.ingest_vote_event(s5, t_days=1.1, is_upvote=False)
    engine.ingest_vote_event(s5, t_days=1.2, is_upvote=False)
    engine.ingest_reopen_event(s5, t_days=2.0)
    score_5 = engine.compute_unified_score(s5, current_age_days=3.0, category_prior=0.3)
    test_vectors["cases"].append({
        "id": "contested_votes_and_reopen",
        "description": "Mixed upvotes, downvotes, and a verified solution failure (reopen).",
        "events": [
            {"type": "vote", "t_days": 1.0, "is_upvote": True},
            {"type": "vote", "t_days": 1.1, "is_upvote": False},
            {"type": "vote", "t_days": 1.2, "is_upvote": False},
            {"type": "reopen", "t_days": 2.0},
        ],
        "expected_final_state": {
            "last_t_days": s5.last_t_days,
            "vote_x_fast": round(s5.vote_x_fast, 8),
            "vote_x_slow": round(s5.vote_x_slow, 8),
            "report_x_fast": round(s5.report_x_fast, 8),
            "report_x_slow": round(s5.report_x_slow, 8),
            "vote_log_lr": round(s5.vote_log_lr, 8),
            "report_log_lr": round(s5.report_log_lr, 8),
            "report_mark_log_lr": round(s5.report_mark_log_lr, 8),
            "vote_dir_log_lr": round(s5.vote_dir_log_lr, 8),
            "reopen_log_lr": round(s5.reopen_log_lr, 8),
            "upvotes": s5.upvotes,
            "downvotes": s5.downvotes,
            "reopens": s5.reopens,
        },
        "evaluations": [
            {
                "current_age_days": 3.0,
                "category_prior": 0.3,
                "unified_log_odds": score_5["unified_log_odds"],
                "posterior_prob_h1": score_5["posterior_prob_h1"],
            }
        ],
    })

    # Case 6: TypeScript Parity Conformance Case
    # Exactly replicates `quietfray/tests/model.test.ts` from MendSeed
    s6 = QuietFrayState()
    ts_events = [
        {"type": "report", "t_days": 0.2, "confidence": 0.8, "diversity": 0.7, "burst_similarity": 0.1},
        {"type": "vote", "t_days": 0.5, "is_upvote": True},
        {"type": "vote", "t_days": 1.2, "is_upvote": True},
        {"type": "report", "t_days": 3.5, "confidence": 0.9, "diversity": 0.6, "burst_similarity": 0.05},
        {"type": "vote", "t_days": 5.0, "is_upvote": False},
        {"type": "vote", "t_days": 7.8, "is_upvote": True},
    ]
    for ev in ts_events:
        if ev["type"] == "report":
            engine.ingest_report_event(s6, ev["t_days"], ev["confidence"], ev["diversity"], ev["burst_similarity"])
        elif ev["type"] == "vote":
            engine.ingest_vote_event(s6, ev["t_days"], ev["is_upvote"])

    pi_ts = (2.0 + 10) / (2.0 + 8.0 + 30)  # y_c=10, n_c=30
    score_6 = engine.compute_unified_score(
        s6,
        current_age_days=10.0,
        category_prior=pi_ts,
        current_horizon_days=12.0,
        has_reopen=False,
    )
    test_vectors["cases"].append({
        "id": "typescript_conformance_parity",
        "description": "Direct conformance benchmark matching MendSeed TypeScript test suite in `model.test.ts`.",
        "events": ts_events,
        "expected_final_state": {
            "last_t_days": s6.last_t_days,
            "vote_x_fast": round(s6.vote_x_fast, 8),
            "vote_x_slow": round(s6.vote_x_slow, 8),
            "report_x_fast": round(s6.report_x_fast, 8),
            "report_x_slow": round(s6.report_x_slow, 8),
            "vote_log_lr": round(s6.vote_log_lr, 8),
            "report_log_lr": round(s6.report_log_lr, 8),
            "report_mark_log_lr": round(s6.report_mark_log_lr, 8),
            "vote_dir_log_lr": round(s6.vote_dir_log_lr, 8),
            "reopen_log_lr": round(s6.reopen_log_lr, 8),
            "upvotes": s6.upvotes,
            "downvotes": s6.downvotes,
            "reopens": s6.reopens,
        },
        "evaluations": [
            {
                "current_age_days": 10.0,
                "category_prior": pi_ts,
                "current_horizon_days": 12.0,
                "has_reopen": False,
                "unified_log_odds": score_6["unified_log_odds"],
                "posterior_prob_h1": score_6["posterior_prob_h1"],
            }
        ],
    })

    return test_vectors


def main():
    fixtures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixtures"))
    os.makedirs(fixtures_dir, exist_ok=True)
    fixtures_path = os.path.join(fixtures_dir, "quietfray_test_vectors_v0.1.json")

    data = build_fixtures()
    with open(fixtures_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(data['cases'])} cross-language test vectors in: {fixtures_path}")


if __name__ == "__main__":
    main()
