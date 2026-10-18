# QuietFray v0.1: Mathematical & Architectural Audit Report

**Date:** 2026-10-18  
**Scope:** `MendSeed/QuietFray` mathematical formulation, reference engine, naming conventions, and cross-language compatibility with the MendSeed ecosystem.

---

## 1. Executive Summary

This audit evaluates the consistency between:
1. The mathematical specifications in `docs/QUIETFRAY_V0.1_SPECIFICATION.md`,
2. The Python reference implementation in `sdk/mendseed_ai/quietflay.py` and canonical `quietfray` package,
3. The expected cross-language platform behavior (`MendSeed/MendSeed` in TypeScript and `MendSeed/ai` in Python).

---

## 2. Naming Conventions & Import Paths

| Component | Historical / Legacy Name | Canonical Public Name | Migration Strategy |
| :--- | :--- | :--- | :--- |
| **Package** | None (`sdk/mendseed_ai`) | `quietfray` | Installable via `pyproject.toml`, top-level `import quietfray` |
| **Engine Class** | `QuietFlayEngineV01`, `QuietFlayEngine` | `QuietFrayEngine` | Aliases provided: `QuietFlayEngine = QuietFrayEngine`, `QuietFlayEngineV01 = QuietFrayEngine` |
| **State Class** | `QuietFlayState` | `QuietFrayState` | Aliased for full backward compatibility |
| **Config Class** | Inlined constants | `QuietFrayConfig` | Exposes hyperparameter overrides, subcriticality checks |
| **Legacy Shim** | `sdk/mendseed_ai/quietflay.py` | Same file preserved | Re-exports all symbols from `quietfray` without code duplication |

---

## 3. Mathematical Equation vs. Implementation Audit

### 3.1 Point-Process Conditional Intensity
$$\lambda_{m,z}(t) = \mu_{c,m} + \alpha_{m,z,f} x_{m,f}(t) + \alpha_{m,z,s} x_{m,s}(t)$$
* **Specification:** Category baseline rate $\mu_{c,m} > 0$ with dual-exponential excitation states.
* **Implementation:** `_advance_channel` evaluates $\lambda_{m,z}$ at $t^-$ immediately before jump.
* **Finding:** When $\mu_{c,m} \ge 10^{-6}$, $\lambda_{m,z} \ge \mu_{\min} > 0$ is guaranteed, enforcing Theorem 2 (Lipschitz continuity).
* **Refinement:** The ratio log $\log(\lambda_1 / \lambda_0)$ in the engine is computed via `math.log(lam_1 / lam_0)` with $\lambda \ge \mu_{\min}$, avoiding any division-by-zero or cancellation.

### 3.2 Inter-Arrival Compensator Integration
$$\Delta I_m(\Delta_j) = \int_{t_{j-1}}^{t_j} [\lambda_{m,1}(t) - \lambda_{m,0}(t)] dt = \sum_{k \in \{f,s\}} (\alpha_{m,1,k} - \alpha_{m,0,k}) x_{m,k}(t_{j-1}^+) \tau_k [1 - \exp(-\Delta_j/\tau_k)]$$
* **Specification:** Baseline rates cancel identically because $\mu_{c,m}$ is equal under $\mathcal{H}_0$ and $\mathcal{H}_1$.
* **Implementation:** Evaluated using `neg_expm1 = -math.expm1(-dt / tau)`.
* **Finding:** Perfectly matches theory. The evaluation of $1 - e^{-u}$ using $-\operatorname{expm1}(-u)$ eliminates catastrophic cancellation for $u \ll 1$ (e.g. sub-second arrival intervals).

### 3.3 Semantic Marks & Beta Densities
$$\ell_{\text{marks}} = \sum_{k=1}^{N_R} \left[ \log \frac{f_1(c_k)}{f_0(c_k)} + \log \frac{g_1(H_k)}{g_0(H_k)} + \log \frac{h_1(s_k^{\text{burst}})}{h_0(s_k^{\text{burst}})} \right]$$
* **Specification:** Evaluates three conjugate Beta likelihood ratios:
  - Confidence $c_k$: $\mathcal{H}_1 \sim \operatorname{Beta}(8, 2)$ vs $\mathcal{H}_0 \sim \operatorname{Beta}(3, 4)$
  - Diversity $H_k$: $\mathcal{H}_1 \sim \operatorname{Beta}(5, 2)$ vs $\mathcal{H}_0 \sim \operatorname{Beta}(1.5, 5)$
  - Burst Sim $s_k^{\text{burst}}$: $\mathcal{H}_1 \sim \operatorname{Beta}(1.5, 8)$ vs $\mathcal{H}_0 \sim \operatorname{Beta}(8, 1.5)$
* **Implementation:** Evaluated via exact gamma function identities `math.lgamma` and `math.log1p(-x)`.
* **Finding:** Input values must be clamped to $[10^{-6}, 1 - 10^{-6}]$ to prevent $\log(0)$ and $\log(1 - 1)$. This clamping is correctly enforced in `_beta_log_lr`.

### 3.4 Vote-Direction Log-Likelihood Updates
$$\Delta \ell_{\text{vote}}^{+} = \log \frac{(a_1 + u)/(a_1 + b_1 + n)}{(a_0 + u)/(a_0 + b_0 + n)}, \qquad \Delta \ell_{\text{vote}}^{-} = \log \frac{(b_1 + d)/(a_1 + b_1 + n)}{(b_0 + d)/(a_0 + b_0 + n)}$$
* **Specification:** Beta-Binomial update with $a_0=2, b_0=2$ and $a_1=7, b_1=3$.
* **Implementation:** Tracks $u$ and $d$ explicitly, computing posterior predictive ratio before incrementing counts.
* **Finding:** Correctly reflects conjugate sequential Bayesian updating.

### 3.5 Survival Function ($\ell_{\text{surv}}$)
$$\ell_{\text{surv}}(a) = \left(\frac{a}{35}\right)^{1.05} - \left(\frac{a}{45}\right)^{0.75}$$
* **Specification:** Weibull shape parameter $k_0 = 1.05 > 1$ (increasing hazard for transient noise) vs $k_1 = 0.75 < 1$ (heavy-tailed survival for chronic issues).
* **Implementation:** Implemented as `((a / 35.0) ** 1.05) - ((a / 45.0) ** 0.75)`.
* **Finding:** Matches specification. Clamped to $a \ge 0.0$ to prevent negative roots.

---

## 4. Discrepancies & Platform Edge Cases

1. **Idle Time Advancement**:
   - In event-driven database updates, `state.last_t_days` reflects the last event arrival.
   - If an issue receives no events for 30 days, its survival term $\ell_{\text{surv}}(a)$ updates based on `current_age_days`, but the Hawkes accumulators $x_f, x_s$ remain at their values from $t_{\text{last}}$ until the next event arrives.
   - **Resolution:** Added `advance_time(state, t_days)` to allow non-event decay and continuous compensator integration when querying current instantaneous states between arrivals.

2. **Temporal Order Clamping**:
   - If out-of-order events arrive ($t_j < t_{\text{last}}$), $dt = \max(0.0, t_j - t_{\text{last}}) = 0.0$.
   - **Resolution:** The engine safely clamps $dt \ge 0$ to avoid numerical NaN. However, platforms must ensure event streams are delivered in order (via monotonically increasing timestamps or outbox offsets). A warning/metric is recommended if $t_j < t_{\text{last}}$.

3. **Subcriticality Enforcement**:
   - Theorem 1 requires $\nu_{m,z} = \alpha_f \tau_f + \alpha_s \tau_s < 1$.
   - **Resolution:** Added automatic subcriticality validation to `QuietFrayConfig`. Attempting to initialize the engine with $\nu \ge 1.0$ raises a `ValueError`.

4. **Synthetic Benchmarking vs. Real-World Claims**:
   - Previous documentation included synthetic metric tables without specifying the generative model or providing an executable script.
   - **Resolution:** Replaced all static claims with an executable benchmark suite (`benchmarks/run_synthetic_benchmark.py`), clear ground-truth labels, exact NDCG calculations against 4 baseline algorithms, and explicit disclaimers regarding synthetic limits.
