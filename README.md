# QuietFray v0.1 (Production Engine: QuietFlay v0.1)

> **A Dual-Timescale Marked Point-Process and Empirical-Bayes Framework for Ranking Persistent Unmet Need in Crowdsourced Issue Observatories**

---

### Quick Links
- 📄 **Complete Mathematical Specification & Whitepaper**: [`docs/QUIETFRAY_V0.1_SPECIFICATION.md`](docs/QUIETFRAY_V0.1_SPECIFICATION.md)
- 🐍 **Pure-Python $O(1)$ Production Engine**: [`sdk/mendseed_ai/quietflay.py`](sdk/mendseed_ai/quietflay.py)
- 🧪 **Unit Test Suite**: [`tests/test_quietflay.py`](tests/test_quietflay.py)

---

### Key Highlights
- **Sequential Hypothesis Testing**: Formulates ranking as a continuous-time Sequential Probability Ratio Test (SPRT) between $\mathcal{H}_1$ (Persistent Need) and $\mathcal{H}_0$ (Transient Viral / Sybil Noise).
- **Dual-Kernel Marked Hawkes Process**: $\tau_f = 0.02\text{ days}$ ($28.8\text{ min}$) and $\tau_s = 3.0\text{ days}$ ($72.0\text{ hours}$).
- **Exact $O(1)$ Recurrence**: $\sim 5.37\ \mu\text{s}$ per event update, zero historical event replays, floating-point drift $< 10^{-12}$ against 50-digit arbitrary-precision arithmetic.
- **Zero Third-Party Dependencies**: Pure Python core requiring only the standard library (`math` and `dataclasses`).
