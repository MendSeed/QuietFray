# QuietFray v0.1 (Production Engine: QuietFlay v0.1)

> **A Dual-Timescale Marked Point-Process and Empirical-Bayes Framework for Ranking Persistent Unmet Need in Crowdsourced Issue Observatories**

Part of the **MendSeed** ecosystem (Community Opportunity & Grievance Observatory).

---

## Key Highlights

- **Sequential Probability Ratio Test (SPRT)**: Formulates issue prioritization as a continuous-time sequential hypothesis test between $\mathcal{H}_1$ (Persistent Need) and $\mathcal{H}_0$ (Transient Viral / Sybil Noise).
- **Dual-Kernel Marked Hawkes Process**: $\tau_f = 0.02\text{ days}$ ($28.8\text{ min}$) and $\tau_s = 3.0\text{ days}$ ($72.0\text{ hours}$).
- **Exact $O(1)$ Recurrence**: $\sim 5.37\ \mu\text{s}$ per event update, zero historical event replays, floating-point drift $< 10^{-12}$ against 50-digit arbitrary-precision arithmetic.
- **Pure Python Standard Library**: Operates strictly on CPU using only Python's standard `math` and `dataclasses`. Zero GPU or neural runtime dependencies.
- **Cross-Language Test Vectors**: Deterministic fixture specifications in `fixtures/quietfray_test_vectors_v0.1.json` for parity verification across Python and TypeScript implementations.

---

## Project Structure

```
MendSeed/QuietFray/
├── pyproject.toml                         # Standard PEP 517/621 installable package
├── src/
│   └── quietfray/                         # Canonical public package
│       ├── __init__.py                    # Public API exports & version 0.1.0
│       ├── engine.py                      # QuietFrayEngine, QuietFrayState, QuietFrayConfig
│       └── baselines.py                   # Classical ranking baselines (RawVotes, Wilson, Decay, EB)
├── sdk/
│   └── mendseed_ai/
│       ├── __init__.py
│       └── quietflay.py                   # Backward-compatibility wrapper for QuietFlay consumers
├── docs/
│   ├── QUIETFRAY_V0.1_SPECIFICATION.md    # Complete mathematical specification and proofs
│   └── AUDIT_REPORT_V0.1.md               # Audit of equations, naming, and platform edge cases
├── fixtures/
│   └── quietfray_test_vectors_v0.1.json   # Reproducible cross-language test vectors
├── scripts/
│   └── generate_fixtures.py               # Fixture generation script
├── benchmarks/
│   ├── run_synthetic_benchmark.py         # Reproducible synthetic benchmark runner
│   └── synthetic_benchmark_results.json   # Generated benchmark metrics (NDCG@K, MRR)
└── tests/
    ├── test_engine.py                     # Comprehensive regression test suite
    ├── test_fixtures.py                   # Cross-language fixture verification test
    ├── test_baselines.py                  # Baseline algorithms test
    ├── test_backward_compat.py            # Backward compatibility verification for QuietFlay imports
    └── test_quietflay.py                  # Legacy test suite
```

---

## Installation & Quickstart

### Install Package Locally
```bash
pip install -e .
```

### Python API Usage
```python
from quietfray import QuietFrayEngine, QuietFrayState

# Initialize engine with default parameters (subcriticality enforced)
engine = QuietFrayEngine()
state = QuietFrayState()

# Ingest a validated community Problem Report (marks in [0, 1])
mark_q = engine.ingest_report_event(
    state,
    t_days=1.0,
    confidence=0.92,       # Qwen structural confidence
    diversity=0.85,        # Lexical/semantic vocabulary entropy
    burst_similarity=0.10, # BGE-small similarity to trailing burst window
)

# Ingest a vote event
engine.ingest_vote_event(state, t_days=1.1, is_upvote=True)

# Compute O(1) unified ranking score
score = engine.compute_unified_score(state, current_age_days=1.5, category_prior=0.3)
print(f"Log-odds: {score['unified_log_odds']}, P(H1): {score['posterior_prob_h1']}")
```

### Backward Compatibility for Existing `QuietFlay` Consumers
Existing consumers importing from `sdk.mendseed_ai.quietflay` are fully supported:
```python
from sdk.mendseed_ai.quietflay import QuietFlayEngineV01, QuietFlayState

engine = QuietFlayEngineV01()
state = QuietFlayState()
```

---

## Verification & Testing

### Running the Test Suite
```bash
py -m pytest tests/
```
All 19 tests verify:
- Initial state and neutral priors
- Fast and slow decay dynamics
- Chronologically ordered event sequences
- Repeated, rapid, and out-of-order event clamping
- Semantic marks ($c_k, H_k, s_k^{\text{burst}}$)
- Downvotes and reopen penalties
- Subcriticality constraint validation ($\nu < 1.0$)
- Finite outputs, numerical stability, and invalid input rejection
- Ranking scenarios (chronic need vs. sudden burst vs. old popular issue)
- Deterministic fixture parity against `fixtures/quietfray_test_vectors_v0.1.json`
- Full backward-compatibility aliases

### Cross-Language Test Vectors Generation
```bash
py scripts/generate_fixtures.py
```

### Running Synthetic Benchmarks
```bash
py benchmarks/run_synthetic_benchmark.py
```
Outputs NDCG@5, NDCG@10, and MRR calculations across 5 controlled synthetic regimes comparing QuietFray v0.1 against Raw Net Votes, Wilson Lower Bound, Exponential Decay, and Empirical Bayes.

---

## Epistemic Boundary Notice
QuietFray v0.1 distinguishes strictly between:
1. **Mathematical Theorems**: Non-explosive Hawkes subcriticality ($\nu < 1$) and cancellation-free floating-point arithmetic (via `expm1`).
2. **Synthetic Benchmarks**: Controlled simulation metrics over generated point-process arrival models.
3. **Real-World Empirical Validation**: Longitudinal human grievance calibration and resistance against adversarial generative paraphrasers, which remain unproven until tested on production traffic.
