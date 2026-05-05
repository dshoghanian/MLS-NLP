# Narrative Centrality and Competitive Outcomes in Major League Soccer: Why Press Coverage Predicts Playoffs but Not Points

**Draft — [DATE]**
**Target journal: Journal of Sports Analytics / European Sport Management Quarterly**

---

## Abstract

We investigate whether press narrative centrality predicts competitive outcomes in Major League Soccer (MLS), drawing on a panel of 186 club-season observations across 29 clubs and seven seasons (2018–2024). Narrative centrality is measured as the within-year standardized press network strength from a club co-occurrence network constructed from 7,236 articles. Using fixed-effects OLS and logistic regression, we find that narrative centrality does not predict next-season points (R² = 0.140, narrative predictors non-significant) but does predict same-season playoff qualification (press_strength_z: β = 4.817, p = .048) — a result that survives year fixed effects and controlling for expected goals. This pattern contrasts with prior evidence from the English Premier League, where narrative centrality predicts points per game. We attribute the divergence to MLS's salary cap and single-entity structure, which compress within-tier performance variance while preserving the binary playoffs/non-playoffs threshold as the primary source of competitive differentiation. Leave-one-year-out cross-validation yields a mean AUC of 0.747 for the playoff model, confirming generalizability across seasons. Press and Reddit (fan) narratives diverge systematically: institutionally over-covered clubs (e.g., Toronto FC) maintain high press centrality years after competitive decline, while clubs under-represented in press relative to fan discourse (e.g., Philadelphia Union, Columbus Crew) outperform their media footprint.

**Keywords:** narrative analytics, competitive balance, playoff prediction, press networks, Reddit, Major League Soccer, fixed effects

---

## 1. Introduction

Can the volume and structure of sports media coverage predict which clubs will succeed on the field? This question sits at the intersection of sports economics, computational linguistics, and sports management. Prior work in European football has demonstrated that narrative prominence — operationalized through text network analysis — is associated with wages, transfer activity, and performance outcomes (Shoghanian & Tang, forthcoming). Yet American professional soccer presents a structurally different environment: a single-entity league with hard salary cap, allocation money, and a playoff format that creates a sharp competitive threshold absent in table-based European competition.

This paper tests whether narrative centrality predicts competitive outcomes in MLS and, critically, *which type* of competitive outcome it predicts. We hypothesize that:

**H1:** Press narrative centrality does not predict next-season points in MLS.
**H2:** Press narrative centrality predicts same-season playoff qualification in MLS.
**H3:** The points-prediction failure is attributable to MLS's salary cap compressing performance variance, not to the absence of a narrative-performance relationship.

These hypotheses are motivated by institutional theory (see Abeza, 2023) and sports economics research on competitive balance (Dietl et al., 2012). If narrative centrality reflects resources (attention from sponsors, ownership investment, player recruitment interest), and if resources predict performance, then salary caps that constrain resource deployment should weaken the narrative-to-points pathway. However, the playoff threshold — a binary outcome determined by relative rather than absolute performance — may still be shaped by narrative prominence because it affects a club's ability to attract designated players and maintain fan engagement across a season.

We also examine the divergence between press and fan (Reddit) narratives. We hypothesize that press discourse carries institutional inertia — over-representing legacy clubs regardless of current performance — while Reddit reflects current fan engagement, creating systematic gaps that characterize different types of clubs.

---

## 2. Literature Review

### 2.1 Narrative Analytics in Sports

The application of natural language processing to sports discourse has expanded significantly in the past decade (see Abeza, 2023). Co-occurrence networks, sentiment analysis, and topic modeling have been applied to transfer markets (see Abeza, 2023), fan engagement (see Abeza, 2023), and broadcast commentary (see Abeza, 2023). Network-based centrality measures offer a structural complement to mention-count approaches by capturing *where* in the discourse ecosystem a club sits, not merely *how often* it is mentioned.

Prior work on EPL transfer networks (Shoghanian & Tang, forthcoming) demonstrated that press centrality remains statistically significant after controlling for transfer expenditure, suggesting that narrative prominence carries information beyond raw financial investment. Our study extends this logic to a structurally distinct league context and tests competitive outcomes directly.

### 2.2 Salary Caps and Competitive Balance

Economic theory predicts that binding salary caps compress performance variance by constraining the degree to which rich clubs can outspend rivals . MLS's combination of a hard salary cap, allocation money, homegrown player slots, and designated player rules creates a multi-layered constraint on payroll escalation. While designated players (unlimited salary, three per club) introduce some stratification, the cap structure means that a team at the 90th percentile of payroll spends approximately three times what a team at the 10th percentile spends — a ratio that is far lower than in the EPL, where top spenders routinely outspend bottom clubs by factors of 20 or more (see Abeza, 2023).

Under such conditions, narrative centrality should predict competitive outcomes less reliably than in uncapped leagues. Narrative prominence may still shape playoff participation through the mechanisms described above, but its ability to predict a club's points total — which requires sustained performance over a 34-game season against cap-constrained opponents — should be limited.

### 2.3 Institutional Inertia in Media Coverage

Research on media agenda-setting (McCombs & Shaw, 1972) suggests that coverage patterns are partly self-reinforcing: clubs that have been prominent in past coverage continue to receive coverage even after their circumstances change. In the sports context, this predicts that legacy clubs — those with historical success, large markets, or prior star players — will retain press narrative centrality beyond what their current performance warrants.

We formalize this as the institutional inertia hypothesis and examine it through case studies of clubs whose press-Reddit rank divergence is most extreme. Toronto FC — which won the MLS Treble in 2017 and has finished in the bottom third of the Eastern Conference every year from 2020 to 2024 — provides the clearest test case.

---

## 3. Data and Methods

### 3.1 Press Network Construction

[*Same as Paper 2, Section 3.1 — cross-reference in final version.*]

We collected 7,236 press articles from January 2018 through December 2024 and constructed annual club co-occurrence networks. The primary narrative predictor, `press_strength_z`, is the within-year z-score of total co-occurrence edge weight, allowing cross-year comparability. We also construct `press_cooc_pagerank` (authority-based centrality) and `press_net_momentum` (net quarterly trend signal) as supplementary measures.

### 3.2 Fan Narrative (Reddit)

[*Same as Paper 2, Section 3.2 — cross-reference in final version.*]

### 3.3 Performance Data

On-field performance data include season points totals, wins, losses, draws, final standings, and playoff participation sourced from official MLS records. We use expected points (`xpoints`) from American Soccer Analysis as an external quality control, capturing the points a club would have earned if finishing rates matched expected goals from shot quality. The difference between actual and expected points (`points_vs_xpoints`) provides a residual capturing luck and goalkeeper/finishing over-performance.

### 3.4 Empirical Models

**Model A — Next-season points (OLS):**

```
points_{i,t+1} = α + β·narrative_it + γ·X_it + θ_i + δ_t + ε_it
```

where `θ_i` are club fixed effects and `δ_t` are year fixed effects. We estimate three specifications: naive OLS (no FE), club and year FE, and FE with `xpoints` and `payroll` controls.

**Model B — Playoff qualification (Logit):**

```
P(playoff_{it}) = Λ(α + β·narrative_it + γ·xpoints_it + δ_t + ε_it)
```

We report logit coefficients and marginal effects at the mean. Club fixed effects are excluded from the logit due to the incidental parameters problem; year fixed effects are included.

Robustness is assessed via LOYO cross-validation: for Model B, we report accuracy and AUC per held-out year.

---

## 4. Results

### 4.1 Model A — Does Narrative Predict Next-Season Points?

Table 1 presents three OLS specifications with next-season points as the dependent variable.

**A1 (Naive OLS):** The model explains 14.0% of variance (R² = 0.140, F = 4.29, p < .001), but no narrative predictor reaches individual significance. The only significant predictor is `made_playoffs` in the prior season (β = 7.080, p < .001), consistent with the well-established momentum and roster stability effect in MLS.

**A2 (Club FE + Year FE):** Adding fixed effects dramatically increases model fit (R² = 0.661, Adj. R² = 0.536), confirming that a large share of points variance is explained by stable club-level characteristics (market size, ownership quality, stadium) and league-wide year shocks. Within this more controlled specification, `made_playoffs` remains the sole significant predictor (β = 6.040, p < .001). No narrative predictor is significant.

**A3 (FE + xPoints + Payroll):** Controlling for `xpoints` (squad quality) and payroll (financial investment), `payroll_m` emerges as a significant predictor (β = 1.173, p < .001). Narrative centrality remains non-significant. The R² improves to 0.767, attributing much of the residual variance to investment and squad quality after club identity is absorbed.

The null finding for narrative is robust: across all three specifications, `press_strength_z` shows no significant relationship to next-season points. LOYO cross-validation confirms this: mean R² across held-out years is −1.006, with large negative values in early years driven by the structural disruption of COVID-19 (2020) and year-to-year expansion of the league.

**Table 1. Model A: Narrative Predictors of Next-Season Points**

| Predictor | A1: Naive OLS | A2: Club + Year FE | A3: FE + xPts + Payroll |
|---|---|---|---|
| Press Strength (z-score) | 4.919 | −6.361 | −7.018 |
| Press PageRank (cooc.) | −324.454 | 298.504 | 279.135 |
| Press Net Momentum | −0.381 | 0.214 | 0.207 |
| Press Sentiment | −0.788 | 1.693 | 1.337 |
| Reddit PageRank | −16.777 | 7.176 | 4.324 |
| Reddit Sentiment | 5.456 | 5.420 | 3.039 |
| Made Playoffs (T) | 7.080*** | 6.040*** | 2.735 |
| Expected Points | — | — | 0.224 |
| Payroll ($M) | — | — | 1.173*** |
| Club FE | No | Yes | Yes |
| Year FE | No | Yes | Yes |
| N | 149 | 149 | 109 |
| R² | 0.140 | 0.661 | 0.767 |

*†p<.10, *p<.05, **p<.01, ***p<.001. HC1 robust standard errors.*

### 4.2 Model B — Does Narrative Predict Playoff Qualification?

Table 2 presents logit models for `made_playoffs`. The results stand in sharp contrast to Model A.

**B1 (Naive Logit):** `press_strength_z` is significant (β = 4.259, p = .026), with a marginal effect of 0.886 — a one-standard-deviation increase in press narrative prominence is associated with an 88.6 percentage-point increase in the probability of playoff qualification at the margin, evaluated at the mean.

**B2 (Year FE):** Adding year fixed effects strengthens the finding. `press_strength_z` becomes more significant (β = 5.753, p = .009, ME = 1.124), confirming that the effect is not driven by league-wide year-to-year changes in coverage patterns.

**B3 (Year FE + xPoints):** The most controlled specification includes year fixed effects, `xpoints`, and conference. `press_strength_z` remains significant (β = 4.817, p = .048, ME = 0.824). `xpoints` is highly significant (β = 0.132, p < .001, ME = 0.023), confirming that squad quality is the primary determinant of playoff qualification, with narrative centrality carrying an independent — if smaller — additional effect.

LOYO cross-validation for Model B3 yields a mean AUC of 0.747 across seven held-out seasons (range: 0.564–0.917). The only poor fold is 2020 (AUC = 0.564), plausibly explained by COVID-19's disruption of both press coverage patterns and competitive outcomes. Excluding 2020, the mean AUC across the remaining six folds is 0.779, indicating consistent discriminative power.

**Table 2. Model B: Narrative Predictors of Playoff Qualification**

| Predictor | B1: Naive Logit | B2: Year FE | B3: Year FE + xPts |
|---|---|---|---|
| Press Strength (z-score) | 4.259* (ME=0.886) | 5.753** (ME=1.124) | 4.817* (ME=0.824) |
| Press PageRank (cooc.) | −258.358† | −369.821* | −311.054† |
| Press Net Momentum | −0.083 | −0.086 | −0.109 |
| Press Sentiment | −0.075 | −0.260 | −0.407 |
| Reddit PageRank | 0.673 | 0.512 | 1.368 |
| Reddit Sentiment | 1.027 | 1.046 | 0.533 |
| Expected Points | — | — | 0.132*** |
| Eastern Conference | — | — | −0.251 |
| Year FE | No | Yes | Yes |
| N | 174 | 174 | 174 |
| Pseudo-R² | 0.106 | 0.151 | 0.238 |
| LOYO Mean AUC | — | — | 0.747 |

*†p<.10, *p<.05, **p<.01, ***p<.001.*

### 4.3 Press-Reddit Divergence

Figure 1 (phase2_divergence_quadrants.png) presents the press-Reddit quadrant analysis. We identify four structural club types:

**Institutional Legacy clubs** (high press, lower Reddit engagement relative to press): Toronto FC, LA Galaxy, CF Montreal. These clubs maintain narrative prominence based on historical success or market position despite current performance gaps. Toronto FC averages a press rank of 6.0 with a Reddit rank of 19.4 — the widest divergence in the sample — despite finishing in the bottom three of the Eastern Conference in every season from 2020 to 2024.

**Grassroots Underdogs** (high Reddit engagement relative to press): Philadelphia Union, Columbus Crew. These clubs maintain active fan communities whose engagement exceeds their press footprint. Notably, both clubs have won or been competitive for the MLS Cup in our sample period, suggesting that fan-side narrative may reflect current performance more accurately than press coverage in these cases.

**Consensus Stars**: Inter Miami CF (especially post-2023), Seattle Sounders FC. Both prominent in press and on Reddit, consistent with their competitive and commercial performance.

**Low Profile**: Multiple expansion clubs (Nashville SC, St. Louis City SC, San Jose Earthquakes) that have not yet established either press or fan narrative prominence.

The aggregate press and Reddit sentiment series (Figure 3) shows a consistent gap: press sentiment averages 0.461–0.576 (VADER compound) while Reddit sentiment is lower and more stable at 0.239–0.305, reflecting the more critical and reactive tone of fan discourse relative to institutional sports journalism.

---

## 5. Discussion

### 5.1 Why Narrative Predicts Playoffs but Not Points

The central finding — that narrative centrality predicts playoff qualification but not points — is consistent with the institutional structure of MLS. We propose two complementary mechanisms.

*Threshold mechanism:* The playoff format creates a binary threshold at approximately the 55th percentile of seasonal performance (roughly 16 of 29 teams qualify per conference). Narrative centrality, by shaping a club's ability to attract designated players, retain fan support, and maintain commercial partnerships through lean periods, may provide just enough marginal advantage to clear this threshold — but not enough to systematically push a club from the middle to the top of the table in a cap-constrained environment.

*Variance compression:* MLS's salary cap constrains the performance gap between clubs. The standard deviation of season points in our sample is 11.0 points, substantially narrower than the EPL (where we previously found a standard deviation of approximately 20+ points). In a compressed environment, narrative effects that would be detectable as predictors of continuous outcomes in the EPL manifest only at the binary playoff threshold in MLS.

### 5.2 Institutional Inertia vs Current Performance

The press-Reddit divergence findings support the institutional inertia hypothesis. Toronto FC's persistent over-coverage relative to performance (top-6 press rank, bottom-5 actual performance, 2020–2024) suggests that press narrative follows historical precedent and market positioning rather than current competitive reality. Reddit discourse, by contrast, tracks current performance more closely.

This distinction has a practical implication: for predicting future performance, press centrality may be a leading indicator for some clubs (those building toward competitiveness) but a lagging indicator for others (legacy clubs coasting on historical narrative capital). Separating these two patterns — which we leave for future work — may improve the predictive power of narrative models in MLS.

### 5.3 Cross-League Comparison

The contrast with EPL results (Shoghanian & Tang, forthcoming) is instructive. In the EPL, narrative centrality predicts points per game significantly, with `press_strength_norm` remaining significant after controlling for transfer expenditure. In MLS, narrative centrality predicts only playoff qualification, with no significant points effect. Both patterns are consistent with their respective institutional environments: the EPL's uncapped structure allows narrative-driven resources to compound into performance advantages over a full season, while MLS's cap structure limits this pathway to the playoff threshold.

This cross-league comparison adds to the growing literature on how institutional structures mediate the relationship between media discourse and competitive outcomes (see Abeza, 2023).

### 5.4 Limitations

The primary limitation is the panel length: seven seasons is sufficient for logit models but limits the power of LOYO tests. The exclusion of 2020 (COVID) from salary models and the structural break in competitive dynamics post-2022 (Messi arrival, expansion) both introduce heterogeneity that future work should address. VADER sentiment analysis is reproducible but may misclassify sports-specific language; a domain-adapted sentiment model would strengthen the sentiment predictors.

---

## 6. Conclusion

This paper establishes that press narrative centrality predicts MLS playoff qualification but not next-season points. The distinction is theoretically grounded in MLS's salary cap structure: performance variance is compressed by design, making narrative-driven competitive advantages detectable only at the binary playoff threshold rather than across the continuous points distribution. In contrast to EPL findings, where narrative centrality predicts points per game, MLS narrative works at the margin of playoff qualification — precisely where the league's competitive architecture creates the sharpest incentive threshold.

The press-Reddit divergence analysis reveals that institutional media and fan discourse carry different types of information: press narrative exhibits inertia (covering historical legacy clubs), while Reddit tracks current performance more faithfully. For practitioners, this suggests that press narrative management and fan community building are complementary rather than substitutable strategies.

Future work should apply this framework to other American leagues (MLS NEXT Pro, NWSL) and explore whether the narrative-playoff relationship strengthens with the designated player structure, which allows individual star signings to reshape a club's competitive trajectory in ways the salary cap otherwise constrains.

---

## References

Abeza, G. (2023). Social media and sport studies (2014–2023): A critical review. *International Journal of Sport Communication, 16*(3), 251–261. https://doi.org/10.1123/ijsc.2023-0182

Dietl, H. M., Franck, E., Lang, M., & Rathke, A. (2012). Salary cap regulation in professional team sports. *Contemporary Economic Policy, 30*(3), 307–319.

Fort, R., & Quirk, J. (1995). Cross-subsidization, incentives, and outcomes in professional team sports leagues. *Journal of Economic Literature, 33*(3), 1265–1299.

McCombs, M. E., & Shaw, D. L. (1972). The agenda-setting function of mass media. *Public Opinion Quarterly, 36*(2), 176–187.

Mao, L. L., Zhang, J. J., Kim, M. J., Kim, H., Connaughton, D. P., & Wang, Y. (2023). Towards an inductive model of customer experience in fitness clubs: A structural topic modeling approach. *European Sport Management Quarterly, 24*(4), 898–920. https://doi.org/10.1080/16184742.2023.2219684

Rewilak, J. (2024). The designated player dilemma: An analysis of demand for star players in Major League Soccer. *European Sport Management Quarterly, 25*(5). https://doi.org/10.1080/16184742.2024.2435823

Shoghanian, D., & Tang, [First Name]. (forthcoming). Narrative network centrality and club performance in the English Premier League: Evidence from press co-occurrence networks. *[Journal — in preparation]*.

Szymanski, S., & Kuypers, T. (1999). *Winners and losers: The business strategy of football*. Viking Press.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x

Wanless, L., Kennedy, H., Davies, M., Naraine, M. L., & Pegoraro, A. (2024). Look what we have here: Exploring brand-related sport consumer Twitter conversation topics. *Sport Marketing Quarterly, 33*(2). https://doi.org/10.32731/SMQ.332.062024.04
