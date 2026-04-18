# Oil Price Shocks & Market Reaction — Methodology & Preliminary Findings
**15.C51 Group Project**  
*Federico Cortesi — April 2026*

---

## Overview

This memo documents the empirical approach taken to answer the core project question:

> *How do equity markets react to oil price shocks, and can we predict that reaction?*

The analysis runs from December 1998 (XLE inception) through April 16, 2026, using daily Brent crude, S&P 500 (^GSPC), and XLE (energy sector ETF) data. All code is in `analysis.py`; all plots are in `plots_event_study/`.

---

## 1. Shock Identification

### Definition
A shock is defined as a day on which Brent crude sets a **new 3-year (756 trading-day) rolling high or low**:

- **Positive shock** (+1): price ≥ 3-year rolling maximum → supply squeeze or demand surge
- **Negative shock** (−1): price ≤ 3-year rolling minimum → supply glut or demand collapse

This is economically cleaner than z-score thresholds on daily returns. A z-score approach labels many consecutive days as "shocks" during volatile regimes (e.g., Gulf War, COVID) — days that are better understood as part of a single shock episode. The 3-year extreme definition identifies *price regime changes*, not daily noise.

### Deduplication
Consecutive shock days within a 10-trading-day window are collapsed into one event, keeping the most extreme price. This gives **63 distinct events** from 1999 to April 2026.

| Type | Count |
|---|---|
| Positive shocks (3y high) | 44 |
| Negative shocks (3y low)  | 19 |
| **Total**                 | **63** |
| Pre-2026 (training)       | 61 |
| 2026 (test)               | 2  |

Notable clusters: Gulf War (1990–91, pre-sample), Asian financial crisis (1998–99), 2008 commodity boom/bust, COVID crash (2020), Ukraine/Russia (2022), US tariff shock (2026).

---

## 2. Event Study Methodology

The event study framework is standard in finance for measuring market reactions to discrete events. The key idea is to **strip out the market's normal expected return** and measure only the *abnormal* component attributable to the shock.

### Estimation Window
For each event at day *t*, we estimate a baseline return model using data from **[t−270, t−30]** — roughly one year of pre-event data, with a 30-day buffer to avoid contaminating the estimate with anticipation effects.

### Baseline Models

| Asset | Model | Formula |
|---|---|---|
| S&P 500 | Constant-mean | $E[R_{SP,\tau}] = \hat{\mu}_{SP}$ (mean over estimation window) |
| XLE | Market model | $E[R_{XLE,\tau}] = \hat{\alpha} + \hat{\beta} \cdot R_{SP,\tau}$ (OLS over estimation window) |

For S&P 500 it does not make sense to use itself as its own benchmark, so we use the simpler mean-adjusted model. For XLE, the market model with S&P as the factor is the standard approach and removes broad market co-movement, isolating the energy-sector-specific abnormal return.

### Abnormal Returns and CARs
$$AR_{i,\tau} = R_{i,\tau} - E[R_{i,\tau}]$$

$$CAR_i(0, h) = \sum_{\tau=0}^{h} AR_{i,\tau}$$

The event window covers **[t−5, t+22]**, so we can observe pre-shock price drift as well as the market's short- and medium-run adjustment.

---

## 3. Event Study Findings

*See `plots_event_study/event_study_car.png` and `car_distributions.png`.*

### Average CAR Paths

**Positive shocks (oil price sets 3-year high):**
- S&P 500 shows a **small negative average CAR** on impact and over the first 5 days — consistent with the view that energy price spikes are a tax on the broader economy.
- XLE shows a **positive abnormal return** on impact, as expected: energy firms benefit directly from higher prices.
- Both effects decay over 22 days.

**Negative shocks (oil price sets 3-year low):**
- S&P 500 CARs are **close to zero and noisy** — negative oil shocks have historically mixed effects on equities (cheap energy helps consumers but signals weak global demand).
- XLE shows **negative CARs**, with more dispersion than the positive shock case.

### Statistical Significance
T-tests of mean CAR against zero (see distribution plot titles):

| | S&P CAR(1d) | S&P CAR(22d) | XLE CAR(1d) | XLE CAR(22d) |
|---|---|---|---|---|
| Positive shocks | p ≈ 0.3 | p ≈ 0.4 | p ≈ 0.1 | p ≈ 0.5 |
| Negative shocks | p ≈ 0.4 | p ≈ 0.5 | p ≈ 0.3 | p ≈ 0.4 |

None of the mean CARs are statistically distinguishable from zero at conventional levels. This is itself a finding: **on average, the market absorbs oil price shocks without a reliably signed abnormal return.** The high cross-sectional dispersion (wide standard deviation bands in the event study plot) suggests the *direction* of each shock matters less than its *context* — the macro regime, oil vol level, and prior market momentum.

---

## 4. Cross-Sectional Model

We model CAR(0→h) as a function of observable shock-day features using three estimators:

**Features (all known at shock day *t*, before the event window outcome):**

| Feature | Interpretation |
|---|---|
| `shock_dir` | +1 / −1 |
| `ret_oil` | Oil return on shock day (magnitude) |
| `sigma_252` | Oil volatility regime (trailing 252d std) |
| `sp_pre_22` | S&P 22d prior return (market momentum context) |
| `dist_from_max` | (price − 3y max) / 3y max — slack from previous extreme |

**Models:** OLS, Ridge (α tuned by 5-fold CV), Lasso (α tuned by 5-fold CV).  
**Training set:** 61 pre-2026 events. **Test set:** 2026 events.

### Performance

| Model | Horizon | CV R² | Train R² |
|---|---|---|---|
| OLS   | 1d  | −1.66 | 0.03 |
| OLS   | 5d  | −29.8 | 0.12 |
| OLS   | 22d | −3.4  | 0.25 |
| Ridge | 1d  | −0.23 | 0.00 |
| Ridge | 5d  | −0.08 | 0.01 |
| **Ridge** | **22d** | **+0.004** | **0.21** |
| Lasso | 1d  | −0.15 | 0.00 |
| Lasso | 5d  | −0.07 | 0.00 |
| Lasso | 22d | −0.06 | 0.17 |

**OLS massively overfits** — with only 61 training events and 5 features, the train/CV gap is enormous. Lasso correctly shrinks all coefficients to zero at 1d and 5d (no signal). Ridge achieves a marginally positive CV R² at 22d, which is the only honest evidence of any predictive content.

**Interpretation:** Short-horizon CARs are essentially unpredictable from shock characteristics alone — consistent with market efficiency at short horizons. At 22 days there is *weak* evidence that features like the volatility regime and prior market momentum shift the conditional distribution of abnormal returns.

---

## 5. Out-of-Sample: 2026 Events

Two 2026 shocks were identified (both positive — oil sets a new 3-year high):

| Date | Oil return | Type |
|---|---|---|
| Mar 13, 2026 | +2.7% | 3-year high |
| Mar 31, 2026 | +4.9% | 3-year high |

*Mar 31 has no 22d CAR available yet (outcome window extends past data cutoff).*

**Mar 13 results:**

| Horizon | Actual CAR | OLS | Ridge | Lasso |
|---|---|---|---|---|
| 1d  | +0.27% | −1.10% | −0.66% | −0.66% |
| 5d  | −2.89% | −0.06% | −0.07% | −0.02% |
| 22d | **+3.75%** | +0.55% | +0.15% | +0.07% |

All models predicted a small negative-to-flat 1d response (directionally plausible — positive oil shock hurts S&P). The actual 22d CAR of +3.75% was far above all model predictions, suggesting either (a) macro factors unrelated to the oil shock dominated over the month, or (b) the sample of 61 training events is too small to calibrate magnitude reliably.

---

## 6. Limitations & Next Steps

1. **Small sample.** 61 training events spanning 25 years is not enough for stable cross-sectional regression. This makes high CV R² structurally impossible and means all quantitative predictions should be treated as directional signals, not precise forecasts.

2. **No macro controls.** The model conditions only on oil-market and equity-market features. Adding recession indicators, Fed funds rate, or global growth proxies (e.g., copper/freight) would likely improve the 22d horizon.

3. **Shock asymmetry.** Positive and negative shocks have very different distributional properties. A direction-stratified model (separate regression for positive vs. negative) may outperform the pooled specification.

4. **XLE analysis.** We computed XLE CARs using the market model but did not yet model XLE CARs cross-sectionally. Given that XLE shows stronger average reactions than S&P (especially for positive shocks), this is worth pursuing.

5. **Demand vs. supply decomposition.** Hamilton (2009) and Kilian (2009) argue that the *cause* of the oil shock matters more than its direction: demand-driven price increases have different equity effects than supply-driven ones. If we can classify events (e.g., using oil futures curve slope, OPEC announcements), that would be a strong extension.
