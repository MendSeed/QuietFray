"""
QuietFlayEngine v0.1: Pure Python Reference Implementation.
Target: sdk/mendseed_ai/quietflay.py
Zero external dependencies (uses standard library math and dataclasses only).
"""
import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class QuietFlayState:
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


class QuietFlayEngineV01:
    """
    Full O(1) Streaming Implementation of QuietFray / QuietFlay v0.1.
    Maintains exact compensator integral differences using -expm1.
    """
    TAU_F: float = 0.02   # 28.8 minutes in days
    TAU_S: float = 3.0    # 72.0 hours in days

    # Hawkes excitation parameters: (alpha_fast, alpha_slow)
    ALPHA_V_H0: Tuple[float, float] = (12.0, 0.025)
    ALPHA_V_H1: Tuple[float, float] = (0.35, 0.160)
    ALPHA_R_H0: Tuple[float, float] = (15.0, 0.020)
    ALPHA_R_H1: Tuple[float, float] = (0.50, 0.180)

    # Beta-Binomial prior parameters for vote direction
    VOTE_A0: float = 2.0
    VOTE_B0: float = 2.0
    VOTE_A1: float = 7.0
    VOTE_B1: float = 3.0

    # Reopen Bernoulli log-likelihood ratio increment
    # log(0.70 / 0.20) = log(3.5)
    DELTA_LLR_REOPEN: float = math.log(0.70 / 0.20)

    def __init__(self, mu_vote: float = 0.1, mu_report: float = 0.05):
        self.mu_vote = max(mu_vote, 1e-6)
        self.mu_report = max(mu_report, 1e-6)

    @staticmethod
    def _beta_log_lr(x: float, a1: float, b1: float, a0: float, b0: float) -> float:
        x_clamped = min(max(x, 1e-6), 1.0 - 1e-6)
        norm_1 = math.lgamma(a1 + b1) - math.lgamma(a1) - math.lgamma(b1)
        norm_0 = math.lgamma(a0 + b0) - math.lgamma(a0) - math.lgamma(b0)
        return (
            (norm_1 - norm_0)
            + (a1 - a0) * math.log(x_clamped)
            + (b1 - b0) * math.log1p(-x_clamped)
        )

    def _advance_channel(
        self,
        x_f: float,
        x_s: float,
        dt: float,
        mu: float,
        alpha_0: Tuple[float, float],
        alpha_1: Tuple[float, float],
        mark_q: float,
    ) -> Tuple[float, float, float]:
        # 1. Exact analytical compensator integral over dt using -expm1
        neg_expm1_f = -math.expm1(-dt / self.TAU_F)
        neg_expm1_s = -math.expm1(-dt / self.TAU_S)

        delta_integral = (
            (alpha_1[0] - alpha_0[0]) * x_f * self.TAU_F * neg_expm1_f
            + (alpha_1[1] - alpha_0[1]) * x_s * self.TAU_S * neg_expm1_s
        )

        # 2. Decay states to t^- immediately prior to event arrival
        x_f_pre = x_f * (1.0 - neg_expm1_f)
        x_s_pre = x_s * (1.0 - neg_expm1_s)

        # 3. Point-process intensity log-likelihood ratio at t^-
        lam_0 = mu + alpha_0[0] * x_f_pre + alpha_0[1] * x_s_pre
        lam_1 = mu + alpha_1[0] * x_f_pre + alpha_1[1] * x_s_pre
        ratio = lam_1 / lam_0
        log_lam_ratio = (
            math.log1p(ratio - 1.0) if ratio >= 1.0 else -math.log1p((1.0 / ratio) - 1.0)
        )

        # 4. Jump state to t^+ by adding bounded mark_q
        return x_f_pre + mark_q, x_s_pre + mark_q, (log_lam_ratio - delta_integral)

    def ingest_report_event(
        self,
        state: QuietFlayState,
        t_days: float,
        confidence: float,
        diversity: float,
        burst_similarity: float,
    ) -> float:
        """
        Updates state in O(1) time upon receiving a validated Problem Report.
        """
        dt = max(0.0, t_days - state.last_t_days)
        mark_q = min(max(confidence * diversity * (1.0 - burst_similarity), 0.0), 1.0)

        state.report_x_fast, state.report_x_slow, d_llr = self._advance_channel(
            state.report_x_fast,
            state.report_x_slow,
            dt,
            self.mu_report,
            self.ALPHA_R_H0,
            self.ALPHA_R_H1,
            mark_q,
        )
        # Decay vote channel synchronously over dt
        neg_expm1_f = -math.expm1(-dt / self.TAU_F)
        neg_expm1_s = -math.expm1(-dt / self.TAU_S)
        state.vote_x_fast *= (1.0 - neg_expm1_f)
        state.vote_x_slow *= (1.0 - neg_expm1_s)

        state.report_log_lr += d_llr
        state.report_mark_log_lr += (
            self._beta_log_lr(confidence, 8.0, 2.0, 3.0, 4.0)
            + self._beta_log_lr(diversity, 5.0, 2.0, 1.5, 5.0)
            + self._beta_log_lr(burst_similarity, 1.5, 8.0, 8.0, 1.5)
        )
        state.last_t_days = t_days
        return mark_q

    def ingest_vote_event(
        self,
        state: QuietFlayState,
        t_days: float,
        is_upvote: bool = True,
    ) -> None:
        """
        Updates state in O(1) time upon receiving a vote.
        """
        dt = max(0.0, t_days - state.last_t_days)
        mark_q = 1.0  # Unit mark for votes

        state.vote_x_fast, state.vote_x_slow, d_llr = self._advance_channel(
            state.vote_x_fast,
            state.vote_x_slow,
            dt,
            self.mu_vote,
            self.ALPHA_V_H0,
            self.ALPHA_V_H1,
            mark_q,
        )
        # Decay report channel synchronously over dt
        neg_expm1_f = -math.expm1(-dt / self.TAU_F)
        neg_expm1_s = -math.expm1(-dt / self.TAU_S)
        state.report_x_fast *= (1.0 - neg_expm1_f)
        state.report_x_slow *= (1.0 - neg_expm1_s)

        state.vote_log_lr += d_llr

        # Sequential Beta-Binomial update
        n = state.upvotes + state.downvotes
        if is_upvote:
            p1 = (self.VOTE_A1 + state.upvotes) / (self.VOTE_A1 + self.VOTE_B1 + n)
            p0 = (self.VOTE_A0 + state.upvotes) / (self.VOTE_A0 + self.VOTE_B0 + n)
            state.vote_dir_log_lr += math.log(p1 / p0)
            state.upvotes += 1
        else:
            p1 = (self.VOTE_B1 + state.downvotes) / (self.VOTE_A1 + self.VOTE_B1 + n)
            p0 = (self.VOTE_B0 + state.downvotes) / (self.VOTE_A0 + self.VOTE_B0 + n)
            state.vote_dir_log_lr += math.log(p1 / p0)
            state.downvotes += 1

        state.last_t_days = t_days

    def ingest_reopen_event(self, state: QuietFlayState, t_days: float) -> None:
        """
        Records a failed resolution / reopen transition.
        """
        state.reopens += 1
        state.reopen_log_lr += self.DELTA_LLR_REOPEN
        state.last_t_days = max(state.last_t_days, t_days)

    def compute_unified_score(
        self,
        state: QuietFlayState,
        current_age_days: float,
        category_prior: float = 0.3,
    ) -> Dict[str, float]:
        """
        Computes the unified log-odds score and posterior probability in O(1) time.
        """
        prior_clamped = min(max(category_prior, 1e-4), 1.0 - 1e-4)
        logit_prior = math.log(prior_clamped / (1.0 - prior_clamped))

        # Weibull duration survival log-likelihood ratio
        a = max(0.0, current_age_days)
        surv_llr = ((a / 35.0) ** 1.05) - ((a / 45.0) ** 0.75)

        u_score = (
            logit_prior
            + state.vote_log_lr
            + state.report_log_lr
            + state.report_mark_log_lr
            + state.vote_dir_log_lr
            + state.reopen_log_lr
            + surv_llr
        )

        # Branch-stabilized sigmoid
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
