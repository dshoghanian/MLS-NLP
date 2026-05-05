# Fixed Effects Regression — Phase 1a & 1b

## Overview

Three model families test whether narrative centrality explains club outcomes after controlling for club-level heterogeneity (club fixed effects) and league-wide shocks (year fixed effects).

**Core narrative predictors across all models:**
- `press_strength_z` — within-year z-score of press co-occurrence edge weight (EPL-equivalent `press_strength_z`)
- `press_cooc_pagerank` — PageRank in the club co-occurrence network
- `press_net_momentum` — net quarterly rising/falling momentum signal
- `reddit_avg_pagerank` — fan-side network centrality
- Sentiment (press and Reddit separately)

All OLS models use HC1 heteroskedasticity-robust standard errors.

---

## Model A — Next-Season Points

**A1** replicates the original OLS with updated regressors (no fixed effects).
**A2** adds club and year fixed effects, absorbing stable team-level differences (market size, stadium capacity, ownership quality) and league-wide year shocks (COVID 2020, Messi 2023).
**A3** further adds `xpoints` (expected points from shot quality) as an external factor normalisation and `payroll` as a financial control.

**Key question:** Does `press_strength_z` or `press_cooc_pagerank` remain significant after absorbing club and year heterogeneity?

### Coefficient Table — Model A (Dep. var: Next-Season Points)

| Feature | A1: Naive OLS | A2: Club FE + Year FE | A3: FE + xPoints + Payroll |
|---|---|---|---|
| Press Strength (z-score) | 4.919 (p=0.532) | -6.361 (p=0.385) | -7.018 (p=0.580) |
| Press PageRank (cooc.) | -324.454 (p=0.599) | 298.504 (p=0.601) | 279.135 (p=0.781) |
| Press Net Momentum | -0.381 (p=0.363) | 0.214 (p=0.495) | 0.207 (p=0.620) |
| Press Sentiment | -0.788 (p=0.705) | 1.692 (p=0.337) | 1.337 (p=0.707) |
| Reddit PageRank | -16.777 (p=0.311) | 7.176 (p=0.499) | 4.324 (p=0.754) |
| Reddit Sentiment | 5.456 (p=0.185) | 5.420 (p=0.382) | 3.039 (p=0.683) |
| Made Playoffs (T) | 7.080*** (p=0.000) | 6.040*** (p=0.000) | 2.735 (p=0.218) |
| Expected Points | —  | —  | 0.224 (p=0.164) |
| Payroll ($M) | —  | —  | 1.173*** (p=0.000) |
| Eastern Conference | —  | —  | -2.191 (p=0.618) |
| Intercept | 52.268* (p=0.013) | 27.672 (p=0.180) | 9.115 (p=0.798) |

---

## Model B — Playoff Qualification (Logit)

Dependent variable: `made_playoffs` (binary).
**B3** (most controlled) includes year FE and `xpoints` to separate narrative prominence from underlying squad quality.

Marginal effects show the change in *probability* of making playoffs for a one-unit increase in each predictor.

### Coefficient Table — Model B (Dep. var: Made Playoffs)

| Feature | B1: Naive Logit (ME) | B2: Logit + Year FE (ME) | B3: Logit + xPoints + Year FE (ME) |
|---|---|---|---|
| Press Strength (z-score) | 4.259* / ME=0.886 | 5.753** / ME=1.124 | 4.817* / ME=0.824 |
| Press PageRank (cooc.) | -258.357 / ME=-53.717 | -369.821* / ME=-72.263 | -311.054 / ME=-53.201 |
| Press Net Momentum | -0.082 / ME=-0.017 | -0.085 / ME=-0.017 | -0.109 / ME=-0.019 |
| Press Sentiment | -0.075 / ME=-0.015 | -0.260 / ME=-0.051 | -0.407 / ME=-0.070 |
| Reddit PageRank | 0.673 / ME=0.140 | 0.512 / ME=0.100 | 1.368 / ME=0.234 |
| Reddit Sentiment | 1.027 / ME=0.214 | 1.046 / ME=0.204 | 0.533 / ME=0.091 |
| Expected Points | —  | —  | 0.132*** / ME=0.023 |
| Eastern Conference | —  | —  | -0.250 / ME=-0.043 |
| Intercept | 9.060 | 14.246* | 6.367 |

---

## Model C — Financial Outcomes

Tests whether narrative centrality independently drives commercial outcomes after controlling for on-field performance and market size.

Dependent variables (all log-transformed except C3):
- **C1** log(Valuation)
- **C2** log(Revenue)
- **C3** Sponsor deal value ($M)
- **C4** log(Payroll)

**Key predictor:** `press_strength_z` — if significant in C1/C2 after controlling for points and attendance, narrative has *independent* commercial value.

### Coefficient Table — Model C (Financial Outcomes)

| Feature | C1: log(Valuation) | C2: log(Revenue) | C3: Sponsor Deal Value ($M) | C4: log(Payroll) |
|---|---|---|---|---|
| Press Strength (z-score) | 0.198*** (p=0.000) | 0.173*** (p=0.000) | 0.498 (p=0.068) | 0.163*** (p=0.000) |
| press_to_reddit_ratio | -0.049 (p=0.240) | -0.049 (p=0.204) | 0.195 (p=0.754) | -0.079** (p=0.004) |
| sentiment_gap | -0.035 (p=0.583) | -0.030 (p=0.610) | -0.195 (p=0.704) | 0.089 (p=0.171) |
| Press Net Momentum | -0.031* (p=0.012) | -0.029** (p=0.009) | -0.124 (p=0.399) | -0.016 (p=0.176) |
| points | 0.002 (p=0.516) | 0.002 (p=0.348) | 0.010 (p=0.740) | -0.005 (p=0.099) |
| Eastern Conference | 0.108* (p=0.024) | 0.079 (p=0.062) | 1.866*** (p=0.000) | 0.019 (p=0.668) |
| log_attendance | 0.142** (p=0.008) | 0.099* (p=0.031) | -0.459 (p=0.286) | 0.132* (p=0.019) |
| Intercept | 4.060*** (p=0.000) | 2.688*** (p=0.000) | 6.888 (p=0.099) | 15.075*** (p=0.000) |

---

## Figures
- `fe_coef_comparison.png` — A1 vs A2 vs A3 coefficient comparison
- `fe_playoff_marginal_effects.png` — Model B3 marginal effects on P(playoff)
- `fe_financial_coefs.png` — Model C narrative predictors across financial outcomes
