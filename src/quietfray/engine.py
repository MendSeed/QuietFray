"""
QuietFray v0.1: Deterministic Continuous-Time Marked Point-Process Ranking Engine.
Part of the MendSeed Opportunity & Grievance Observatory.

Zero external dependencies: relies exclusively on standard library `math` and `dataclasses`.
Runs entirely on CPU without GPU or neural model runtime requirements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class QuietFrayConfig:
    """
    Configuration parameters and subcriticality bounds for QuietFray v0.1.
    All temporal constants are denominated in days.
    """
    # Relaxation timescales (days)
    tau_f: float = 0.02   # 28.8 minutes
    tau_s: float = 3.0    # 72.0 hours

    # Hawkes excitation coefficients: (alpha_fast, alpha_slow) in days^-1
    alpha_v_h0: Tuple[float, float] = (12.0, 0.025)
    alpha_v_h1: Tuple[float, float] = (0.35, 0.160)
    alpha_r_h0: Tuple[float, float] = (15.0, 0.020)
    alpha_r_h1: Tuple[float, float] = (0.50, 0.180)

    # Immigrant baseline arrival rates (days^-1) - aligned with MendSeed platform defaults
    mu_vote: float = 0.05
    mu_report: float = 0.02
    mu_min_floor: float = 1e-6

    # Beta-Binomial prior parameters for vote direction
    vote_a0: float = 2.0
    vote_b0: float = 2.0
    vote_a1: float = 7.0
    vote_b1: float = 3.0

    # Beta densities for report marks: (a1, b1, a0, b0)
    beta_confidence: Tuple[float, float, float, float] = (8.0, 2.0, 3.0, 4.0)
    beta_diversity: Tuple[float, float, float, float] = (5.0, 2.0, 1.5, 5.0)
    beta_burst_sim: Tuple[float, float, float, float] = (1.5, 8.0, 8.0, 1.5)

    # Weibull survival parameters: (shape_k, scale_theta)
    weibull_h0: Tuple[float, float] = (1.05, 35.0)
    weibull_h1: Tuple[float, float] = (0.75, 45.0)

    # Reopening Bernoulli probabilities: (p_h1, p_h0)
    reopen_p1: float = 0.70
    reopen_p0: float = 0.20

    def __post_init__(self) -> None:
        self.validate_bounds()

    def validate_bounds(self) -> None:
        """Enforces physical constraints and Theorem 1 subcriticality branching ratios."""
        if self.tau_f <= 0.0 or self.tau_s <= 0.0:
            raise ValueError(f"Timescales must be strictly positive: tau_f={self.tau_f}, tau_s={self.tau_s}")
        if self.tau_f >= self.tau_s:
            raise ValueError(f"Fast timescale tau_f must be strictly less than slow timescale tau_s: {self.tau_f} >= {self.tau_s}")

        ratios = {
            "votes_h0": self.alpha_v_h0[0] * self.tau_f + self.alpha_v_h0[1] * self.tau_s,
            "votes_h1": self.alpha_v_h1[0] * self.tau_f + self.alpha_v_h1[1] * self.tau_s,
            "reports_h0": self.alpha_r_h0[0] * self.tau_f + self.alpha_r_h0[1] * self.tau_s,
            "reports_h1": self.alpha_r_h1[0] * self.tau_f + self.alpha_r_h1[1] * self.tau_s,
        }
        for channel, ratio in ratios.items():
            if ratio >= 1.0:
                raise ValueError(
                    f"Subcriticality violation in channel '{channel}': "
                    f"spectral branching ratio nu = {ratio:.4f} >= 1.0 (process would be explosive)."
                )

    @property
    def branching_ratios(self) -> Dict[str, float]:
        """Returns the calculated branching ratios nu_m,z for each channel and hypothesis."""
        return {
            "votes_h0": self.alpha_v_h0[0] * self.tau_f + self.alpha_v_h0[1] * self.tau_s,
            "votes_h1": self.alpha_v_h1[0] * self.tau_f + self.alpha_v_h1[1] * self.tau_s,
            "reports_h0": self.alpha_r_h0[0] * self.tau_f + self.alpha_r_h0[1] * self.tau_s,
            "reports_h1": self.alpha_r_h1[0] * self.tau_f + self.alpha_r_h1[1] * self.tau_s,
        }


@dataclass
class QuietFrayState:
    """
    O(1) Memory Footprint Streaming State per Canonical Issue.
    Maintains exact cumulative log-likelihood evidence and exponential decay accumulators.
    """
    last_t_days: float = 0.0
    vote_x_fast: float = 0.0
    vote_x_slow: float = 0.0
    report_x_fast: float = 0.0
    report_x_slow: float = 0.0
    vote_log_lr: float = 0.0
    report_log_lr: float = 0.0
    report_mark_log_lr: float = 0.0
    vote_dir_log_lr: float = 0.0
    reopen_log_lr: float = 0.0
    upvotes: int = 0
    downvotes: int = 0
    reopens: int = 0


class QuietFrayEngine:
    """
    QuietFray v0.1 Streaming Evidence Engine.
    Executes exact O(1) state updates per event in ~5 microseconds.
    """

    def __init__(self, config: Optional[QuietFrayConfig] = None, mu_vote: Optional[float] = None, mu_report: Optional[float] = None):
        self.config = config or QuietFrayConfig()
        if mu_vote is not None:
            self.config.mu_vote = max(mu_vote, self.config.mu_min_floor)
        if mu_report is not None:
            self.config.mu_report = max(mu_report, self.config.mu_min_floor)

        self.mu_vote = max(self.config.mu_vote, self.config.mu_min_floor)
        self.mu_report = max(self.config.mu_report, self.config.mu_min_floor)
        self.reopen_log_lr_delta = math.log(self.config.reopen_p1 / self.config.reopen_p0)

    @staticmethod
    def _beta_log_lr(x: float, a1: float, b1: float, a0: float, b0: float) -> float:
        """
        Exact log-density ratio log(Beta(x; a1, b1) / Beta(x; a0, b0))
        computed with machine precision via lgamma and log1p.
        """
        x_clamped = min(max(x, 1e-6), 1.0 - 1e-6)
        norm_1 = math.lgamma(a1 + b1) - math.lgamma(a1) - math.lgamma(b1)
        norm_0 = math.lgamma(a0 + b0) - math.lgamma(a0) - math.lgamma(b0)
        return (
            (norm_1 - norm_0)
            + (a1 - a0) * math.log(x_clamped)
            + (b1 - b0) * math.log1p(-x_clamped)
        )

    def _advance_decay_and_compensators(self, state: QuietFrayState, dt: float) -> Tuple[float, float]:
        """
        Advances decay and continuous compensator integration for BOTH channels over interval dt.
        Returns the negative expm1 factors (neg_expm1_f, neg_expm1_s).
        """
        if dt <= 0.0:
            return 0.0, 0.0

        neg_expm1_f = -math.expm1(-dt / self.config.tau_f)
        neg_expm1_s = -math.expm1(-dt / self.config.tau_s)

        # Vote channel compensator integral difference & decay
        delta_int_v = (
            (self.config.alpha_v_h1[0] - self.config.alpha_v_h0[0]) * state.vote_x_fast * self.config.tau_f * neg_expm1_f
            + (self.config.alpha_v_h1[1] - self.config.alpha_v_h0[1]) * state.vote_x_slow * self.config.tau_s * neg_expm1_s
        )
        state.vote_x_fast *= (1.0 - neg_expm1_f)
        state.vote_x_slow *= (1.0 - neg_expm1_s)
        state.vote_log_lr -= delta_int_v

        # Report channel compensator integral difference & decay
        delta_int_r = (
            (self.config.alpha_r_h1[0] - self.config.alpha_r_h0[0]) * state.report_x_fast * self.config.tau_f * neg_expm1_f
            + (self.config.alpha_r_h1[1] - self.config.alpha_r_h0[1]) * state.report_x_slow * self.config.tau_s * neg_expm1_s
        )
        state.report_x_fast *= (1.0 - neg_expm1_f)
        state.report_x_slow *= (1.0 - neg_expm1_s)
        state.report_log_lr -= delta_int_r

        return neg_expm1_f, neg_expm1_s

    def advance_time(self, state: QuietFrayState, t_days: float) -> None:
        """
        Advances state time to `t_days` without firing an event.
        Decays Hawkes accumulators and accrues idle compensator integral differences.
        """
        if t_days <= state.last_t_days:
            return
        dt = t_days - state.last_t_days
        self._advance_decay_and_compensators(state, dt)
        state.last_t_days = t_days

    def ingest_report_event(
        self,
        state: QuietFrayState,
        t_days: float,
        confidence: float,
        diversity: float,
        burst_similarity: float,
    ) -> float:
        """
        Updates state upon receiving a validated Problem Report.
        Validates mark bounds and computes composite mark q in [0, 1].
        """
        if math.isnan(t_days) or math.isnan(confidence) or math.isnan(diversity) or math.isnan(burst_similarity):
            raise ValueError("Input parameters cannot be NaN")

        dt = max(0.0, t_days - state.last_t_days)
        c = min(max(confidence, 0.0), 1.0)
        h = min(max(diversity, 0.0), 1.0)
        s_burst = min(max(burst_similarity, 0.0), 1.0)

        mark_q = min(max(c * h * (1.0 - s_burst), 0.0), 1.0)

        # 1. Advance both channels' continuous compensators and decay states to t^-
        self._advance_decay_and_compensators(state, dt)

        # 2. Evaluate report intensity jump ratio at t^-
        lam_0 = self.mu_report + self.config.alpha_r_h0[0] * state.report_x_fast + self.config.alpha_r_h0[1] * state.report_x_slow
        lam_1 = self.mu_report + self.config.alpha_r_h1[0] * state.report_x_fast + self.config.alpha_r_h1[1] * state.report_x_slow
        state.report_log_lr += math.log(lam_1 / lam_0)

        # 3. Add mark jump to report accumulators at t^+
        state.report_x_fast += mark_q
        state.report_x_slow += mark_q

        # 4. Evaluate semantic mark log-evidence
        b_conf = self.config.beta_confidence
        b_div = self.config.beta_diversity
        b_burst = self.config.beta_burst_sim

        state.report_mark_log_lr += (
            self._beta_log_lr(c, b_conf[0], b_conf[1], b_conf[2], b_conf[3])
            + self._beta_log_lr(h, b_div[0], b_div[1], b_div[2], b_div[3])
            + self._beta_log_lr(s_burst, b_burst[0], b_burst[1], b_burst[2], b_burst[3])
        )
        state.last_t_days = max(state.last_t_days, t_days)
        return mark_q

    def ingest_vote_event(
        self,
        state: QuietFrayState,
        t_days: float,
        is_upvote: bool = True,
    ) -> None:
        """
        Updates state in O(1) time upon receiving an upvote or downvote.
        """
        if math.isnan(t_days):
            raise ValueError("t_days cannot be NaN")

        dt = max(0.0, t_days - state.last_t_days)
        mark_q = 1.0

        # 1. Advance both channels' continuous compensators and decay states to t^-
        self._advance_decay_and_compensators(state, dt)

        # 2. Evaluate vote intensity jump ratio at t^-
        lam_0 = self.mu_vote + self.config.alpha_v_h0[0] * state.vote_x_fast + self.config.alpha_v_h0[1] * state.vote_x_slow
        lam_1 = self.mu_vote + self.config.alpha_v_h1[0] * state.vote_x_fast + self.config.alpha_v_h1[1] * state.vote_x_slow
        state.vote_log_lr += math.log(lam_1 / lam_0)

        # 3. Add mark jump to vote accumulators at t^+
        state.vote_x_fast += mark_q
        state.vote_x_slow += mark_q

        # 4. Sequential Beta-Binomial update
        n = state.upvotes + state.downvotes
        if is_upvote:
            p1 = (self.config.vote_a1 + state.upvotes) / (self.config.vote_a1 + self.config.vote_b1 + n)
            p0 = (self.config.vote_a0 + state.upvotes) / (self.config.vote_a0 + self.config.vote_b0 + n)
            state.vote_dir_log_lr += math.log(p1 / p0)
            state.upvotes += 1
        else:
            p1 = (self.config.vote_b1 + state.downvotes) / (self.config.vote_a1 + self.config.vote_b1 + n)
            p0 = (self.config.vote_b0 + state.downvotes) / (self.config.vote_a0 + self.config.vote_b0 + n)
            state.vote_dir_log_lr += math.log(p1 / p0)
            state.downvotes += 1

        state.last_t_days = max(state.last_t_days, t_days)

    def ingest_reopen_event(self, state: QuietFrayState, t_days: float) -> None:
        """
        Records a failed resolution attempt or reopen transition.
        Increments reopen counter and adds delta log-likelihood ratio.
        """
        if math.isnan(t_days):
            raise ValueError("t_days cannot be NaN")
        state.reopens += 1
        state.reopen_log_lr += self.reopen_log_lr_delta
        state.last_t_days = max(state.last_t_days, t_days)

    def compute_unified_score(
        self,
        state: QuietFrayState,
        current_age_days: float,
        category_prior: float = 0.3,
        current_horizon_days: Optional[float] = None,
        has_reopen: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Computes the unified log-odds score and posterior probability in O(1) time.
        Supports optional `current_horizon_days` to integrate tail compensators and
        `has_reopen` boolean flag for full parity with MendSeed platform clients.
        """
        if math.isnan(current_age_days) or math.isnan(category_prior):
            raise ValueError("Score inputs cannot be NaN")

        prior_clamped = min(max(category_prior, 1e-4), 1.0 - 1e-4)
        logit_prior = math.log(prior_clamped / (1.0 - prior_clamped))

        # Weibull duration survival log-likelihood ratio
        a = max(0.0, current_age_days)
        k0, th0 = self.config.weibull_h0
        k1, th1 = self.config.weibull_h1
        surv_llr = ((a / th0) ** k0) - ((a / th1) ** k1)

        # Reopen signal
        if has_reopen is not None:
            reopen_llr = (
                math.log(self.config.reopen_p1 / self.config.reopen_p0)
                if has_reopen
                else math.log((1.0 - self.config.reopen_p1) / (1.0 - self.config.reopen_p0))
            )
        else:
            reopen_llr = state.reopen_log_lr

        # Tail integral if horizon extends beyond last recorded event
        tail_llr_v = 0.0
        tail_llr_r = 0.0
        if current_horizon_days is not None and current_horizon_days > state.last_t_days:
            dt = current_horizon_days - state.last_t_days
            neg_expm1_f = -math.expm1(-dt / self.config.tau_f)
            neg_expm1_s = -math.expm1(-dt / self.config.tau_s)
            delta_int_v = (
                (self.config.alpha_v_h1[0] - self.config.alpha_v_h0[0]) * state.vote_x_fast * self.config.tau_f * neg_expm1_f
                + (self.config.alpha_v_h1[1] - self.config.alpha_v_h0[1]) * state.vote_x_slow * self.config.tau_s * neg_expm1_s
            )
            delta_int_r = (
                (self.config.alpha_r_h1[0] - self.config.alpha_r_h0[0]) * state.report_x_fast * self.config.tau_f * neg_expm1_f
                + (self.config.alpha_r_h1[1] - self.config.alpha_r_h0[1]) * state.report_x_slow * self.config.tau_s * neg_expm1_s
            )
            tail_llr_v = -delta_int_v
            tail_llr_r = -delta_int_r

        u_score = (
            logit_prior
            + (state.vote_log_lr + tail_llr_v)
            + (state.report_log_lr + tail_llr_r)
            + state.report_mark_log_lr
            + state.vote_dir_log_lr
            + reopen_llr
            + surv_llr
        )

        # Branch-stabilized sigmoid to prevent floating-point overflow
        if u_score >= 0.0:
            prob_h1 = 1.0 / (1.0 + math.exp(-u_score))
        else:
            exp_u = math.exp(u_score)
            prob_h1 = exp_u / (1.0 + exp_u)

        return {
            "unified_log_odds": round(u_score, 8),
            "posterior_prob_h1": round(prob_h1, 8),
            "vote_fast_state": round(state.vote_x_fast, 6),
            "vote_slow_state": round(state.vote_x_slow, 6),
            "report_fast_state": round(state.report_x_fast, 6),
            "report_slow_state": round(state.report_x_slow, 6),
            "survival_llr": round(surv_llr, 6),
        }


# Backward compatibility aliases for consumers migrated from QuietFlay
QuietFlayEngine = QuietFrayEngine
QuietFlayEngineV01 = QuietFrayEngine
QuietFlayState = QuietFrayState
