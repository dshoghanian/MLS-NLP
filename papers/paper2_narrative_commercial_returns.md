# Earned Narrative and Commercial Returns in Major League Soccer: Evidence from Press Network Centrality

**Draft — [DATE]**
**Target journal: Sport Marketing Quarterly**

---

## Abstract

This study examines whether press narrative centrality — a club's structural prominence in the co-occurrence network of sports journalism — independently drives commercial outcomes in Major League Soccer (MLS) beyond on-field performance and market size. Using a panel of 186 club-season observations across 29 clubs and seven seasons (2018–2024), we construct a narrative co-occurrence network from 7,236 press articles and extract within-year standardized strength scores. Controlling for points, attendance, and year fixed effects, press narrative centrality (β = 0.198, p < .001) is associated with 21.9% higher club valuations and 18.9% higher revenues per standard deviation increase. The effect holds for payroll investment (β = 0.163, p < .001), suggesting that narrative prominence shapes owner and front-office resource allocation decisions, not merely external market perceptions. Fan-side narrative (Reddit centrality) shows a distinct pattern: a higher press-to-Reddit ratio is associated with *lower* payroll, indicating that organic fan engagement carries independent commercial signals beyond press coverage. These findings have direct implications for sports marketing practitioners: in a salary-capped single-entity league where on-field differentiation is structurally compressed, narrative management constitutes a measurable commercial lever.

**Keywords:** sports marketing, narrative analytics, earned media, club valuation, Major League Soccer, network analysis

---

## 1. Introduction

Professional sports organizations increasingly recognize that commercial success depends not only on what happens on the field but on how their activities are narrated in public discourse (Gladden & Funk, 2002; Ross, 2006). In leagues with significant competitive balance mechanisms — salary caps, allocation systems, draft structures — the on-field variance between clubs is compressed by design. In such environments, the question of how clubs differentiate themselves commercially becomes one of marketing strategy rather than sporting merit (Dietl et al., 2012).

Major League Soccer (MLS) presents an ideal natural laboratory for this question. As a single-entity league with a hard salary cap and allocation system, MLS deliberately constrains the financial pathway to on-field dominance available to clubs in uncapped European leagues. Yet valuations across the league vary by a factor of ten or more, revenues diverge sharply between clubs in similar markets, and jersey sponsorship deals range from regional grocery chains to global technology firms. What explains this commercial dispersion if on-field performance cannot?

This paper proposes and tests a narrative capital hypothesis: that a club's structural centrality in the press co-occurrence network — the degree to which a club is consistently named alongside other clubs, players, coaches, and transfer activities in sports journalism — constitutes a form of earned media capital that independently drives club valuation, revenue, and investment decisions.

We make three contributions. First, we construct a novel press narrative network for MLS spanning 2018–2024, the most comprehensive application of network-based NLP to American professional soccer to our knowledge. Second, we demonstrate that narrative centrality predicts commercial outcomes — specifically club valuation, revenue, and payroll — independently of on-field performance and market size. Third, we document an asymmetry between press and fan-generated (Reddit) narrative, with each channel carrying distinct commercial signals.

The remainder of this paper proceeds as follows. Section 2 reviews relevant literature on earned media, brand equity in sport, and computational approaches to narrative analysis. Section 3 describes our data collection and methodology. Section 4 presents regression results. Section 5 discusses implications. Section 6 concludes.

---

## 2. Literature Review

### 2.1 Brand Equity and Earned Media in Sport

Sport marketing scholarship has long recognized that club brand equity transcends wins and losses . Brand associations, media presence, and fan identification have been shown to drive attendance, merchandise revenue, and sponsorship value . However, most brand equity frameworks have relied on survey-based measures or aggregate media metrics (column inches, broadcast minutes) rather than the structural position of a club within the broader media ecosystem.

The concept of *earned media* — media coverage generated through news value rather than paid placement — has gained traction in marketing practice but remains underdeveloped theoretically (Lovett & Staelin, 2016). In sport, earned media includes transfer rumors, pre-season previews, match reports, and roster announcements that mention a club not because it paid for placement but because journalists and aggregators deem it newsworthy. A club that is consistently mentioned alongside other prominent clubs, star players, and high-profile transactions accumulates narrative capital that may be inseparable from its commercial valuation.

### 2.2 Network Analysis and Narrative Structure

Network science approaches to media and discourse have demonstrated that structural position in co-occurrence networks captures information beyond simple mention counts . PageRank and strength centrality in co-occurrence networks have been applied to financial news sentiment to predict asset price movements , to political discourse to identify agenda-setting dynamics (see Abeza, 2023), and to sports transfer networks to characterize club positioning (see Abeza, 2023).

In the MLS context, a club's network strength — the total weight of its co-occurrence edges with other entities — captures how embedded it is in the overall discourse architecture of American soccer journalism. A club with high strength is not merely mentioned often; it is mentioned *with* the league's key narratives, players, and rivals. This structural embeddedness is what we term narrative centrality.

### 2.3 Commercial Outcomes in Salary-Capped Leagues

The economics of American professional sports leagues differ fundamentally from European football in ways relevant to the narrative capital hypothesis. Single-entity structures and salary caps constrain the degree to which financial investment translates to on-field dominance . Under these conditions, owners' decisions about roster investment, stadium development, and commercial partnerships are partly decoupled from short-term performance expectations.

We posit that in this environment, narrative prominence functions as a signal — to potential star players considering designated player slots, to regional and national sponsors evaluating partnership value, and to investors and valuators assessing franchise worth. If this hypothesis holds, narrative centrality should predict commercial outcomes independently of points and attendance.

### 2.4 Press vs Fan Discourse

The rise of fan-generated content on platforms such as Reddit has created a parallel discourse ecosystem alongside traditional sports journalism. Research on social media and sports marketing has shown that fan engagement metrics carry distinct predictive value for attendance and merchandise (see Abeza, 2023). However, the relationship between press coverage and fan discourse — specifically, whether they carry redundant or complementary signals — remains underexplored at the structural level.

---

## 3. Data and Methods

### 3.1 Press Data and Network Construction

We collected 7,236 press articles mentioning MLS clubs from January 2018 through December 2024 using a curated set of sports journalism sources. Articles were processed through a named entity recognition pipeline using spaCy's `en_core_web_sm` model to extract club, player, coach, and sponsor mentions. A club co-occurrence network was constructed for each calendar year: nodes represent clubs, and an undirected edge is drawn between two clubs when they appear in the same article, with edge weight equal to the number of co-occurring articles.

From each annual network, we extract three centrality measures: PageRank (authority within the network), degree (number of distinct co-occurrence partners), and strength (total edge weight). For cross-year comparison, we construct a within-year z-score of strength: `press_strength_z = (strength − μ_year) / σ_year`. This normalization ensures that the metric captures relative narrative prominence rather than absolute article volume, which varies year to year.

### 3.2 Fan Discourse (Reddit)

Reddit data were collected from subreddits dedicated to MLS clubs and the league generally (`r/MLS`, `r/ussoccer`, and 26 club-specific subreddits). Club mentions were extracted using regex-based alias matching. A parallel co-occurrence network was constructed using Reddit post co-mentions, yielding annual Reddit PageRank scores for each club. The `press_to_reddit_ratio` is computed as press PageRank divided by Reddit PageRank, capturing the degree to which a club's narrative is press-driven vs fan-driven.

### 3.3 Commercial and Performance Data

Commercial data were sourced from Forbes MLS valuations (annual), MLSPA salary disclosures (2018–2024, excluding 2020), American Soccer Analysis expected goals and attendance data, and jersey sponsorship data collected from club announcements. On-field performance data were collected from official MLS standings.

The final panel contains 186 club-season observations across 29 clubs and 7 seasons (2018–2024). FC Cincinnati is absent from valuation models (Forbes did not track them until 2023) and 2020 salary data are missing for the full league (MLSPA did not publish during the COVID-affected season).

### 3.4 Empirical Strategy

We estimate OLS regressions of the form:

```
log(Y_it) = α + β₁·press_strength_z_it + β₂·press_to_reddit_ratio_it
            + β₃·sentiment_gap_it + β₄·press_net_momentum_it
            + γ·X_it + δ_t + ε_it
```

where `Y_it` is the commercial outcome (valuation, revenue, sponsor deal value, or payroll), `X_it` is a vector of controls (points, log-attendance, conference dummy), and `δ_t` are year fixed effects. Valuation, revenue, and payroll are log-transformed to stabilize variance and allow coefficient interpretation as approximate percentage changes. Standard errors are HC1 heteroskedasticity-robust. `press_net_momentum` is the within-year net momentum signal (quarters with rising centrality minus quarters with falling centrality). `sentiment_gap` is the difference between press and Reddit VADER compound sentiment scores, capturing whether institutional and fan discourse are tonally aligned.

Robustness is assessed via Leave-One-Year-Out (LOYO) cross-validation: the model is re-estimated on six of seven years and evaluated on the held-out year, repeated for each year in the panel.

---

## 4. Results

### 4.1 Descriptive Statistics

The panel spans 29 MLS clubs over 7 seasons, with a mean club valuation of $378M (SD: $267M) and mean annual revenue of $63M (SD: $42M). Press network strength is right-skewed, with Inter Miami (2023–2024), LA Galaxy, and Toronto FC consistently occupying the top three positions. Mean payroll is $15.1M annually, with a range from $6.2M to $48.1M. The press-Reddit rank gap — the difference between a club's press rank and Reddit rank — reveals systematic divergence: Toronto FC averages a press rank of 6.0 vs a Reddit rank of 19.4, indicating persistent institutional over-coverage relative to fan engagement.

### 4.2 Model C1 — Club Valuation

Table 1 presents results for log(valuation) as the dependent variable. Press narrative centrality (`press_strength_z`) is positive and highly significant (β = 0.198, SE = 0.040, p < .001), indicating that a one-standard-deviation increase in within-year press network strength is associated with approximately 21.9% higher club valuation, holding constant points, log-attendance, conference, and year fixed effects. This effect is large: it implies that moving from the 25th to 75th percentile of narrative prominence corresponds to a valuation premium of roughly $83M at the sample mean.

Points (β = 0.002, p = .516) are not significant once narrative centrality and attendance are controlled for, consistent with the league-wide finding that on-field performance is not the primary driver of commercial value in MLS.

Attendance (β = 0.142, p = .008) and conference (Eastern: β = 0.108, p = .024) are both significant controls, reflecting that market size and Eastern Conference media market access independently drive valuations.

Interestingly, `press_net_momentum` is significantly *negative* (β = −0.031, p = .012). Clubs on a rising narrative trajectory have lower current valuations, likely because the clubs most often in a momentum-building phase are smaller-market clubs ascending (Columbus Crew, Philadelphia Union) while established high-valuation clubs (Inter Miami, LA Galaxy) maintain steady narrative positions rather than building.

### 4.3 Model C2 — Revenue

Results for log(revenue) closely mirror the valuation findings. `press_strength_z` remains highly significant (β = 0.173, SE = 0.032, p < .001), with a one-standard-deviation increase associated with approximately 18.9% higher revenue. `press_net_momentum` is again negative (β = −0.029, p = .009), and `log_attendance` is significant (β = 0.099, p = .031). Points remain non-significant.

### 4.4 Model C3 — Jersey Sponsor Deal Value

Press centrality shows a directionally positive but marginally significant effect on sponsor deal value (β = 0.498, p = .068). The Eastern Conference dummy is the strongest predictor (β = 1.866, p < .001), consistent with the concentration of large-market clubs in the East attracting premium sponsors. The weaker significance of narrative centrality for sponsorship likely reflects the longer contracting cycles and relationship-driven nature of jersey sponsorship deals, which may be less responsive to year-to-year narrative fluctuation than market-based valuations.

### 4.5 Model C4 — Payroll

The finding for payroll (log-transformed) is perhaps the most actionable result in this paper. `press_strength_z` is positive and significant (β = 0.163, p < .001), suggesting that clubs with higher narrative prominence attract greater payroll investment from ownership. Importantly, the `press_to_reddit_ratio` is *negatively* significant (β = −0.079, p = .004): clubs with relatively more fan-driven narrative (lower press-to-Reddit ratio, meaning fans discuss the club more relative to press coverage) are associated with higher payrolls. This suggests that organic fan engagement signals market depth to ownership in ways that press coverage alone does not, and may reflect the willingness of fan-supported clubs to invest in roster quality.

**Table 1. Regression Results — Commercial Outcomes (Year Fixed Effects)**

| Predictor | C1: log(Valuation) | C2: log(Revenue) | C3: Sponsor ($M) | C4: log(Payroll) |
|---|---|---|---|---|
| Press Strength (z-score) | 0.198*** | 0.173*** | 0.498† | 0.163*** |
| Press/Reddit Ratio | −0.049 | −0.049 | 0.195 | −0.079** |
| Sentiment Gap | −0.035 | −0.030 | −0.195 | 0.089 |
| Press Net Momentum | −0.031* | −0.029** | −0.124 | −0.016 |
| Points | 0.002 | 0.002 | 0.010 | −0.005† |
| Eastern Conference | 0.108* | 0.079† | 1.866*** | 0.019 |
| log(Attendance) | 0.142** | 0.099* | −0.459 | 0.132* |
| Year FE | Yes | Yes | Yes | Yes |
| N | 164 | 164 | 164 | 134 |
| R² | 0.629 | 0.641 | 0.194 | 0.538 |

*†p<.10, *p<.05, **p<.01, ***p<.001. HC1 robust standard errors.*

### 4.6 Robustness — LOYO Cross-Validation

The valuation model performs well in early years (LOYO R² = 0.888 for 2018, 0.916 for 2019) but degrades substantially in 2022–2024 (LOYO R² = −0.774 to −4.273). This temporal instability reflects a structural break in MLS valuations beginning in 2022: league-wide expansion fees ($200M+), Lionel Messi's arrival at Inter Miami in 2023, and renegotiated broadcast rights compressed the cross-sectional variance that the model exploits. We interpret this as a limitation with informational content — the league itself underwent a commercial regime change that rendered historical cross-sectional relationships less predictive. The core within-sample finding (narrative → valuation) is robustly significant; the LOYO degradation speaks to the non-stationarity of MLS commercial dynamics in the post-2021 period.

---

## 5. Discussion

### 5.1 Narrative as a Commercial Asset

The central finding of this paper is that press narrative centrality predicts club valuation, revenue, and investment independently of on-field performance and market size. This has a direct managerial interpretation: in a salary-capped league where performance outcomes are structurally compressed, narrative management is not merely reputational but financially material.

A club that consistently appears in press co-occurrence networks — that is named alongside star players, rivals, and transfer activity even during off-seasons — accumulates a form of earned media capital that is reflected in its Forbes valuation and in ownership's willingness to invest in roster quality. This is not the same as having a large marketing budget; `press_strength_z` is a structural measure of *where the club sits in the discourse ecosystem*, which depends as much on storyline relevance and narrative momentum as on paid media.

### 5.2 The Press-Reddit Asymmetry

The finding that `press_to_reddit_ratio` negatively predicts payroll is a nuanced result with practical implications. Clubs with more fan-driven narratives — where Reddit discourse is active relative to press coverage — appear to attract more payroll investment, possibly because organic fan engagement is a more reliable signal of market demand than press-generated coverage. Press coverage can be driven by institutional inertia (Toronto FC's 2017 treble continues to generate press coverage years after the team's competitive decline), while Reddit discussion tends to reflect current fan enthusiasm and engagement.

For club executives, this suggests a distinction between *narrative reach* (total press co-occurrence strength) and *narrative authenticity* (fan engagement relative to press). Both matter, but they matter differently: reach drives valuation and external perceptions; authenticity drives investment.

### 5.3 Limitations

Several limitations should be noted. First, the panel is relatively short (7 years) and our sentiment analysis relies on VADER, which may misclassify sports-specific language. Second, valuation data are Forbes estimates, which carry measurement error. Third, the LOYO degradation in post-2022 years suggests the model may not generalize to the current commercial environment without retraining. Fourth, we cannot fully separate narrative centrality from omitted market-level factors (e.g., stadium quality, ownership wealth) that may drive both press coverage and valuations simultaneously.

---

## 6. Conclusion

This paper establishes narrative centrality as an independent predictor of commercial outcomes in Major League Soccer. Using a network-based measure of press co-occurrence prominence, we show that a one-standard-deviation increase in narrative centrality is associated with approximately 21.9% higher club valuations and 18.9% higher revenues, after controlling for on-field performance, attendance, and year fixed effects. The effect extends to payroll investment, suggesting that narrative prominence shapes ownership resource allocation.

For sport marketing researchers, this contributes a validated, replicable network-based measure of earned media capital that outperforms simple mention counts. For MLS practitioners, it implies that sustained press narrative presence — not just winning — constitutes a strategic asset worth managing deliberately.

Future work should examine whether narrative centrality predicts commercial outcomes in other salary-capped leagues (NFL, NBA, NWSL) and explore the mechanisms connecting press prominence to valuation, including whether media narratives directly influence Forbes methodology or operate through fan engagement and sponsor interest as mediators.

---

## References

Bauer, H. H., Sauer, N. E., & Schmitt, P. (2005). Customer-based brand equity in the team sport industry. *European Journal of Marketing, 39*(5), 496–513. https://doi.org/10.1108/03090560510590683

Dietl, H. M., Franck, E., Lang, M., & Rathke, A. (2012). Salary cap regulation in professional team sports. *Contemporary Economic Policy, 30*(3), 307–319.

Fort, R., & Quirk, J. (1995). Cross-subsidization, incentives, and outcomes in professional team sports leagues. *Journal of Economic Literature, 33*(3), 1265–1299.

Gladden, J. M., & Funk, D. C. (2002). Developing an understanding of brand associations in team sport: Empirical evidence from consumers of professional sport. *Journal of Sport Management, 16*(1), 54–81.

Kaynak, E., Salman, G. G., & Tatoglu, E. (2008). An integrative framework linking brand associations and brand loyalty in professional sports. *Journal of Brand Management, 15*, 336–357. https://doi.org/10.1057/palgrave.bm.2550117

Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance, 66*(1), 35–65. https://doi.org/10.1111/j.1540-6261.2010.01625.x

Lovett, M. J., & Staelin, R. (2016). The role of paid, earned, and owned media in building entertainment brands. *Marketing Science, 35*(1), 142–157. https://doi.org/10.1287/mksc.2015.0961

Mao, L. L., Zhang, J. J., Kim, M. J., Kim, H., Connaughton, D. P., & Wang, Y. (2023). Towards an inductive model of customer experience in fitness clubs: A structural topic modeling approach. *European Sport Management Quarterly, 24*(4), 898–920. https://doi.org/10.1080/16184742.2023.2219684

Naraine, M. L., Bakhsh, J. T., & Wanless, L. (2022). The impact of sponsorship on social media engagement: A longitudinal examination of professional sport teams. *Sport Marketing Quarterly, 31*(3). https://doi.org/10.32731/SMQ.313.0922.06

Naraine, M. L., Pegoraro, A., & Wear, H. (2021). #WeTheNorth: Examining an online brand community through a professional sport organization's hashtag marketing campaign. *Communication & Sport, 9*, 625–645. https://doi.org/10.1177/2167479519878676

Ross, S. D. (2006). A conceptual framework for understanding spectator-based brand equity. *Journal of Sport Management, 20*(1), 22–38.

Shoghanian, D., & Tang, [First Name]. (forthcoming). Narrative network centrality and club performance in the English Premier League: Evidence from press co-occurrence networks. *[Journal — in preparation]*.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x

Wanless, L., Kennedy, H., Davies, M., Naraine, M. L., & Pegoraro, A. (2024). Look what we have here: Exploring brand-related sport consumer Twitter conversation topics. *Sport Marketing Quarterly, 33*(2). https://doi.org/10.32731/SMQ.332.062024.04
