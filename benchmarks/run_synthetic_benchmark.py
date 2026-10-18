"""
Synthetic Benchmark Suite for QuietFray v0.1 vs Classical Baselines.
Evaluates ranking performance (NDCG@5, NDCG@10, MRR) over generated scenarios
with explicit ground-truth relevance labels (y_i in {0, 1}).

DISCLAIMER:
This benchmark measures performance under controlled synthetic generative assumptions.
It does NOT constitute real-world empirical validation on live human community data.

Usage:
    py benchmarks/run_synthetic_benchmark.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quietfray import QuietFrayConfig, QuietFrayEngine, QuietFrayState
from quietfray.baselines import (
    raw_net_votes,
    wilson_lower_bound,
    exponential_recency_decay,
    empirical_bayes_vote_score,
)


@dataclass
class SyntheticIssue:
    issue_id: str
    ground_truth_label: int  # 1 = True Persistent Need, 0 = Transient / Sybil / Noise
    category: str
    age_days: float
    reports: List[Tuple[float, float, float, float]]  # (t_days, confidence, diversity, burst_sim)
    votes: List[Tuple[float, int]]  # (t_days, direction: +1 or -1)
    reopens: List[float]  # [t_days, ...]


def dcg_at_k(relevance_scores: List[int], k: int = 10) -> float:
    """Calculates Discounted Cumulative Gain at rank K."""
    score = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        score += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
    return score


def ndcg_at_k(ranked_labels: List[int], k: int = 10) -> float:
    """Calculates Normalized Discounted Cumulative Gain at rank K."""
    dcg = dcg_at_k(ranked_labels, k)
    ideal_labels = sorted(ranked_labels, reverse=True)
    idcg = dcg_at_k(ideal_labels, k)
    return (dcg / idcg) if idcg > 0.0 else 0.0


def reciprocal_rank(ranked_labels: List[int]) -> float:
    """Returns Mean Reciprocal Rank (1 / rank of first relevant item)."""
    for i, rel in enumerate(ranked_labels):
        if rel > 0:
            return 1.0 / (i + 1.0)
    return 0.0


def generate_regime_dataset(regime_name: str, num_issues: int = 100, seed: int = 42) -> List[SyntheticIssue]:
    """
    Generates synthetic issues for a controlled adversarial regime.
    Each issue has an explicit ground_truth_label: 1 (relevant) or 0 (irrelevant).
    """
    rng = random.Random(seed)
    issues: List[SyntheticIssue] = []

    for idx in range(num_issues):
        is_relevant = (idx % 2 == 0)
        issue_id = f"{regime_name}_issue_{idx:03d}"
        reports: List[Tuple[float, float, float, float]] = []
        votes: List[Tuple[float, int]] = []
        reopens: List[float] = []

        if regime_name == "Regime_A_Weak_Signal":
            # True issues have low-volume but steady, high-quality reports;
            # Irrelevant issues have low-information sporadic noise.
            age_days = rng.uniform(7.0, 14.0)
            if is_relevant:
                num_reps = rng.randint(4, 8)
                for _ in range(num_reps):
                    t = rng.uniform(0.5, age_days)
                    c = rng.uniform(0.80, 0.95)
                    h = rng.uniform(0.70, 0.90)
                    s_burst = rng.uniform(0.05, 0.20)
                    reports.append((t, c, h, s_burst))
                for r in reports:
                    votes.append((r[0] + rng.uniform(0.01, 0.1), 1))
            else:
                num_reps = rng.randint(1, 4)
                for _ in range(num_reps):
                    t = rng.uniform(0.5, age_days)
                    c = rng.uniform(0.30, 0.60)
                    h = rng.uniform(0.15, 0.40)
                    s_burst = rng.uniform(0.20, 0.50)
                    reports.append((t, c, h, s_burst))
                num_votes = rng.randint(1, 10)
                for _ in range(num_votes):
                    votes.append((rng.uniform(0.5, age_days), rng.choice([1, -1])))

        elif regime_name == "Regime_B_Sybil_Spike":
            # True issues have persistent steady reports over 10 days;
            # Irrelevant issues have a sudden massive raid of 50 reports within 15 mins.
            age_days = 10.0
            if is_relevant:
                for day in range(1, 10):
                    reports.append((float(day), 0.85, 0.80, 0.10))
                    votes.append((float(day) + 0.1, 1))
                    votes.append((float(day) + 0.2, 1))
            else:
                t_spike = 9.8
                for s in range(50):
                    t = t_spike + s * 0.0001
                    reports.append((t, 0.55, 0.15, 0.95))
                    votes.append((t + 0.00001, 1))

        elif regime_name == "Regime_C_Legacy_Monopoly":
            # True issues are emerging chronic problems created 10 days ago with steady recent reports;
            # Irrelevant issues are legacy issues created 180 days ago with 200 ancient votes and zero recent activity.
            if is_relevant:
                age_days = 10.0
                for d in range(1, 10):
                    reports.append((float(d), 0.88, 0.85, 0.12))
                    votes.append((float(d) + 0.1, 1))
            else:
                age_days = 180.0
                # 200 votes long ago (between day 1 and day 20)
                for v in range(200):
                    t = rng.uniform(1.0, 20.0)
                    votes.append((t, 1))

        elif regime_name == "Regime_D_Contested_Controversy":
            # True issues have strong positive consensus;
            # Irrelevant issues have polarized flame wars (e.g. 50 upvotes, 48 downvotes).
            age_days = 12.0
            if is_relevant:
                for d in range(1, 12, 2):
                    reports.append((float(d), 0.85, 0.80, 0.10))
                    votes.append((float(d) + 0.1, 1))
                    votes.append((float(d) + 0.2, 1))
            else:
                for _ in range(40):
                    t = rng.uniform(1.0, 12.0)
                    votes.append((t, 1))
                for _ in range(38):
                    t = rng.uniform(1.0, 12.0)
                    votes.append((t, -1))

        else:  # Regime_E_Composite_Mixed
            age_days = rng.uniform(5.0, 30.0)
            if is_relevant:
                for d in range(int(age_days)):
                    if rng.random() > 0.3:
                        reports.append((float(d), rng.uniform(0.8, 0.95), rng.uniform(0.7, 0.9), rng.uniform(0.05, 0.2)))
                        votes.append((float(d) + 0.1, 1))
            else:
                if rng.random() > 0.5:
                    # Sybil burst
                    t_sp = age_days - 0.2
                    for s in range(30):
                        reports.append((t_sp + s * 0.0002, 0.5, 0.1, 0.92))
                        votes.append((t_sp + s * 0.0002, 1))
                else:
                    # Sporadic noise
                    for _ in range(rng.randint(2, 6)):
                        votes.append((rng.uniform(0.0, age_days), rng.choice([1, -1])))

        # Sort events by timestamp
        reports.sort(key=lambda x: x[0])
        votes.sort(key=lambda x: x[0])
        reopens.sort()

        issues.append(SyntheticIssue(
            issue_id=issue_id,
            ground_truth_label=1 if is_relevant else 0,
            category="general",
            age_days=age_days,
            reports=reports,
            votes=votes,
            reopens=reopens,
        ))

    return issues


def score_dataset_with_quietfray(engine: QuietFrayEngine, issues: List[SyntheticIssue]) -> List[Tuple[float, int]]:
    """Evaluates issues using QuietFrayEngine, returning list of (score, ground_truth_label)."""
    scored = []
    for issue in issues:
        state = QuietFrayState()
        # Merge all events chronologically
        timeline = []
        for r in issue.reports:
            timeline.append((r[0], "report", r))
        for v in issue.votes:
            timeline.append((v[0], "vote", v))
        for ro in issue.reopens:
            timeline.append((ro, "reopen", ro))

        timeline.sort(key=lambda x: x[0])

        for t, ev_type, data in timeline:
            if ev_type == "report":
                _, conf, div, burst = data
                engine.ingest_report_event(state, t_days=t, confidence=conf, diversity=div, burst_similarity=burst)
            elif ev_type == "vote":
                _, direction = data
                engine.ingest_vote_event(state, t_days=t, is_upvote=(direction > 0))
            elif ev_type == "reopen":
                engine.ingest_reopen_event(state, t_days=t)

        score_dict = engine.compute_unified_score(state, current_age_days=issue.age_days)
        scored.append((score_dict["unified_log_odds"], issue.ground_truth_label))
    return scored


def score_dataset_with_baselines(issues: List[SyntheticIssue]) -> Dict[str, List[Tuple[float, int]]]:
    """Evaluates issues using the 4 defined classical baselines."""
    results = {
        "Raw_Net_Votes": [],
        "Wilson_Bound": [],
        "Exponential_Decay": [],
        "Empirical_Bayes": [],
    }

    for issue in issues:
        up = sum(1 for _, d in issue.votes if d > 0)
        down = sum(1 for _, d in issue.votes if d < 0)

        s_raw = raw_net_votes(up, down)
        s_wilson = wilson_lower_bound(up, down)
        s_decay = exponential_recency_decay(issue.votes, current_time_days=issue.age_days, tau_days=3.0)
        s_eb = empirical_bayes_vote_score(up, down, alpha=2.0, beta=2.0)

        results["Raw_Net_Votes"].append((s_raw, issue.ground_truth_label))
        results["Wilson_Bound"].append((s_wilson, issue.ground_truth_label))
        results["Exponential_Decay"].append((s_decay, issue.ground_truth_label))
        results["Empirical_Bayes"].append((s_eb, issue.ground_truth_label))

    return results


def run_benchmark():
    engine = QuietFrayEngine()
    regimes = [
        "Regime_A_Weak_Signal",
        "Regime_B_Sybil_Spike",
        "Regime_C_Legacy_Monopoly",
        "Regime_D_Contested_Controversy",
        "Regime_E_Composite_Mixed",
    ]

    benchmark_output = {
        "benchmark_type": "Synthetic Controlled Simulation",
        "model_version": "QuietFray v0.1",
        "notice": (
            "These metrics represent synthetic benchmarks over simulated arrival distributions. "
            "They do NOT constitute real-world empirical validation or longitudinal human trials."
        ),
        "regimes": {},
    }

    print("=" * 86)
    print("QuietFray v0.1 Synthetic Benchmark: Ranking Persistent Need vs Baselines")
    print("=" * 86)
    print(f"{'Regime':<30} | {'Model':<18} | {'NDCG@5':<8} | {'NDCG@10':<8} | {'MRR':<8}")
    print("-" * 86)

    for regime in regimes:
        issues = generate_regime_dataset(regime, num_issues=120, seed=1337)

        # 1. QuietFray
        qf_scored = score_dataset_with_quietfray(engine, issues)
        qf_scored.sort(key=lambda x: x[0], reverse=True)
        qf_labels = [label for _, label in qf_scored]

        qf_ndcg5 = ndcg_at_k(qf_labels, 5)
        qf_ndcg10 = ndcg_at_k(qf_labels, 10)
        qf_mrr = reciprocal_rank(qf_labels)

        regime_results = {
            "QuietFray_v0.1": {
                "NDCG@5": round(qf_ndcg5, 4),
                "NDCG@10": round(qf_ndcg10, 4),
                "MRR": round(qf_mrr, 4),
            }
        }
        print(f"{regime:<30} | {'QuietFray v0.1':<18} | {qf_ndcg5:<8.4f} | {qf_ndcg10:<8.4f} | {qf_mrr:<8.4f}")

        # 2. Baselines
        baseline_scored = score_dataset_with_baselines(issues)
        for b_name, b_pairs in baseline_scored.items():
            b_pairs.sort(key=lambda x: x[0], reverse=True)
            b_labels = [label for _, label in b_pairs]
            b_ndcg5 = ndcg_at_k(b_labels, 5)
            b_ndcg10 = ndcg_at_k(b_labels, 10)
            b_mrr = reciprocal_rank(b_labels)

            regime_results[b_name] = {
                "NDCG@5": round(b_ndcg5, 4),
                "NDCG@10": round(b_ndcg10, 4),
                "MRR": round(b_mrr, 4),
            }
            print(f"{'':<30} | {b_name:<18} | {b_ndcg5:<8.4f} | {b_ndcg10:<8.4f} | {b_mrr:<8.4f}")

        print("-" * 86)
        benchmark_output["regimes"][regime] = regime_results

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "synthetic_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2)

    print(f"\nBenchmark calculations saved to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
