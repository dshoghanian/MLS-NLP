# When Media and Fans Disagree: Press-Reddit Narrative Divergence in Major League Soccer

**Draft — [DATE]**
**Format: Research Note / Communication**
**Target journal: International Journal of Sport Communication**

---

## Abstract

This research note documents systematic divergence between institutional press narratives and fan-generated discourse (Reddit) in Major League Soccer (MLS), using a panel of 186 club-season observations from 2018 to 2024. We construct parallel press and Reddit co-occurrence networks and classify clubs into four quadrant types based on relative press and Reddit narrative prominence. Institutional Legacy clubs — those maintaining high press centrality despite competitive decline — are exemplified by Toronto FC (press rank average: 6.0; Reddit rank average: 19.4; competitive finish average: 24th of 29). Grassroots Underdog clubs — those with fan engagement exceeding their press footprint — include Philadelphia Union and Columbus Crew, both of which outperformed their narrative standing competitively. Granger causality tests on the aggregate annual sentiment series (N=7) are inconclusive due to series length, but both series are non-stationary, suggesting co-movement rather than lead-lag causation at annual frequency. The press-Reddit rank gap shows no predictive relationship with next-year valuation change (r = −0.012), indicating that divergence characterizes club types but does not independently drive commercial outcomes. We discuss implications for sport communication researchers and club communication practitioners.

**Keywords:** press coverage, Reddit, fan discourse, narrative divergence, media agenda-setting, MLS, sport communication

---

## 1. Background and Motivation

The proliferation of fan-generated content on social media platforms has created a second, parallel discourse ecosystem alongside traditional sports journalism. Where press coverage is shaped by editorial gatekeeping, newsworthiness norms, and institutional relationships, fan communities on platforms such as Reddit produce discourse driven by current engagement, emotional investment, and collective sense-making.

Research on social media and sports has documented that fan-generated content predicts attendance (see Abeza, 2023), influences purchase behavior (see Abeza, 2023), and shapes club brand perceptions (see Abeza, 2023). However, the *structural relationship* between press and fan narratives — specifically, whether they track the same clubs, cover the same events, and move together over time — has received less attention.

This note addresses three questions: (1) Do press and fan narratives systematically diverge in MLS, and if so, which clubs show the largest gaps? (2) Do press and Reddit sentiment series exhibit a lead-lag relationship (Granger causality) at the league level? (3) Does press-Reddit divergence predict future commercial outcomes?

---

## 2. Data and Method

### 2.1 Press Network

From 7,236 MLS press articles (2018–2024), we construct annual club co-occurrence networks and extract PageRank centrality. Club-year narrative ranks are assigned within each season (rank 1 = highest PageRank).

### 2.2 Reddit Network

Reddit posts were collected from `r/MLS`, `r/ussoccer`, and 26 club-specific subreddits. Co-occurrence networks were constructed from club co-mentions in posts, and Reddit PageRank ranks were computed within each season.

### 2.3 Divergence Measure

The press-Reddit rank gap = press rank − Reddit rank. A positive gap indicates the club is more prominent on Reddit than in press (grassroots underdog); a negative gap indicates the club is more prominent in press than on Reddit (institutional legacy).

### 2.4 Granger Causality

We aggregate club-level sentiment (VADER compound score) to annual league means, yielding a 7-observation series for press and Reddit. Augmented Dickey-Fuller (ADF) tests assess stationarity. Granger causality tests at lag 1 examine whether past press sentiment predicts current Reddit sentiment and vice versa.

---

## 3. Results

### 3.1 Quadrant Classification

Figure 1 plots average press rank against average Reddit rank for all 29 clubs. We classify clubs into four quadrants relative to the sample median:

**Institutional Legacy** (high press, low Reddit relative prominence): Toronto FC, LA Galaxy, CF Montreal. These clubs carry historical narrative capital — Toronto's 2017 treble, LA Galaxy's Beckham/Ibrahimović era, CF Montreal's longtime transfer activity — that sustains press coverage independent of current performance.

**Grassroots Underdogs** (high Reddit, lower press prominence): Philadelphia Union, Columbus Crew. Both clubs have active and analytically sophisticated fan communities. Philadelphia Union finished as MLS Cup Runner-up (2022) and maintained a top-5 xPoints ranking across multiple seasons, yet averaged a press rank of only 9 — below clubs with inferior on-field records.

**Consensus Stars**: Inter Miami CF (driven by the 2023 Messi arrival and its aftermath), Seattle Sounders FC (three MLS Cup appearances in the sample period). Both prominent in press and fan discourse, with close alignment between channels.

**Low Profile**: Nashville SC, San Jose Earthquakes, St. Louis City SC (expansion), Colorado Rapids. Below-median in both channels.

### 3.2 Case Study Clubs

**Toronto FC** presents the clearest case of institutional inertia. Press rank averaged 6.0 across 2018–2024, driven by legacy brand recognition from the 2017 treble and the Toronto media market. Reddit rank averaged 19.4. Despite this press prominence, Toronto FC finished in the bottom three of the Eastern Conference in every season from 2020 to 2024. The press narrative maintained historical associations (star player arrivals, coaching changes) even as competitive performance deteriorated. The gap widened monotonically after 2019: from a gap of +1 in 2018 to a gap of +21 in 2022 and 2023.

**CF Montreal** is the inverse case in one dimension: despite achieving a conference final appearance (2021) and competitive seasons, press rank averaged only 3.9 (very prominent, but this largely reflects MLSPA-related coverage, transfer activity, and the Montreal media market in both English and French). Reddit rank averaged 12.0. The apparent paradox resolves when the aggregator bias in the press pipeline is considered: a single multi-club aggregator source inflated CF Montreal's co-occurrence weight in early pipeline versions and was subsequently filtered.

**Charlotte FC** illustrates expansion over-hype: the highest press-Reddit gap among expansion clubs at launch, with a press rank of 22 against a performance rank of 8. Expansion narratives generate press coverage (new stadium, new market, designated player signings) that does not translate to immediate fan community depth or competitive performance.

**Philadelphia Union** exemplifies the grassroots underdog: a consistently competitive team (top-5 xPoints, MLS Cup finalist) with a Reddit rank averaging 6.0 but a press rank averaging 9. The Union's tactical sophistication and youth development model generate fan community engagement that outpaces press interest in a smaller market.

### 3.3 Aggregate Sentiment Series

Press sentiment (mean VADER compound: 0.461–0.576) consistently exceeds Reddit sentiment (0.239–0.305), reflecting the more critical and reactive tone of fan communities relative to institutional sports journalism. Both series show year-to-year fluctuation without strong trend: press sentiment dipped in 2020 (COVID disruption, μ = 0.356) and peaked in 2024 (μ = 0.576); Reddit sentiment shows less variance (range: 0.239–0.305).

### 3.4 Granger Causality

ADF tests find both series non-stationary (press: p = .125; Reddit: p = .256) at annual frequency. Granger tests at lag 1 are non-significant in both directions (press → Reddit: F = 0.158, p = .718; Reddit → press: F = 0.335, p = .603).

We interpret this null result carefully. With N = 7 annual observations, statistical power is negligible. The non-stationarity suggests co-movement rather than lead-lag causation: press and Reddit sentiment may respond to the same underlying events (league expansion, star arrivals, COVID) rather than one leading the other. The monthly series (N = 84, available through the quarterly centrality data) would provide adequate power for Granger testing and is recommended for future work.

### 3.5 Press-Reddit Gap and Commercial Outcomes

The correlation between press-Reddit rank gap and next-year valuation percentage change is r = −0.012. Divergence between channels characterizes structural club types but does not independently predict commercial value creation. This suggests that the gap is a *diagnostic* rather than a *predictive* tool: it describes the nature of a club's narrative position rather than forecasting its financial trajectory.

---

## 4. Discussion

### 4.1 Two Ecosystems, Different Information

The press and Reddit narrative ecosystems in MLS carry complementary information. Press coverage tracks institutional narratives — historical legacy, market size, designated player transactions, coaching appointments — with relative inertia. Reddit tracks current engagement, tactical discourse, and emotional investment with more sensitivity to recent performance.

For sport communication practitioners, this asymmetry has direct implications. Press narrative is more amenable to management through traditional media relations — press conferences, transfer announcements, strategic storytelling — because of its institutional structure and editorial accessibility. Reddit discourse is harder to manage but more diagnostic: persistent negative divergence (fans more critical than press) may signal brewing supporter dissatisfaction that eventually surfaces in attendance, merchandise, or broader commercial metrics.

### 4.2 Institutional Inertia as a Market Inefficiency

The Toronto FC case suggests that institutional press coverage can create a narrative market inefficiency: a club is priced, in media terms, at its historical reputation rather than its current competitive reality. In financial markets, analogous inefficiencies correct as investors update (Tetlock, 2007). In sports media markets, the correction appears slower, possibly because narrative inertia serves the interests of both clubs (positive coverage) and media outlets (familiar, established brands attract readers).

Whether this inertia eventually corrects — and whether the correction precedes or follows competitive decline — is a question this dataset begins to address. Toronto FC's press rank has declined from 5 in 2018 to a still-elevated 4 in 2024, suggesting slow adjustment even after seven years of non-playoff finishes.

### 4.3 Limitations and Future Directions

The primary limitation is Reddit coverage completeness: `r/cfmontreal` was not collected, meaning CF Montreal's Reddit metrics reflect only incidental mentions from `r/MLS` and `r/ussoccer`. This likely underestimates CF Montreal's actual fan discourse and may inflate their apparent press-Reddit gap.

The Granger analysis is underpowered at annual frequency. Future work using the monthly or quarterly time series available in this dataset could test lead-lag relationships with adequate power and examine whether press sentiment leads Reddit sentiment around specific events (transfer windows, playoff races, star signings).

---

## 5. Conclusion

This research note documents systematic press-Reddit divergence in MLS and characterizes four structural club types based on the relative prominence of institutional and fan-generated narratives. Institutional Legacy clubs (Toronto FC, LA Galaxy) maintain press centrality beyond their competitive reality; Grassroots Underdogs (Philadelphia Union, Columbus Crew) are under-covered relative to competitive performance and fan engagement. Granger causality tests are inconclusive at annual frequency, and press-Reddit divergence does not predict commercial outcomes in isolation.

These findings contribute a structural, network-based characterization of the press-fan narrative relationship in American professional soccer, with implications for sport communication research on agenda-setting, fan community development, and club brand management.

---

## References

Abeza, G. (2023). Social media and sport studies (2014–2023): A critical review. *International Journal of Sport Communication, 16*(3), 251–261. https://doi.org/10.1123/ijsc.2023-0182

McCombs, M. E., & Shaw, D. L. (1972). The agenda-setting function of mass media. *Public Opinion Quarterly, 36*(2), 176–187.

Naraine, M. L., Bakhsh, J. T., & Wanless, L. (2022). The impact of sponsorship on social media engagement: A longitudinal examination of professional sport teams. *Sport Marketing Quarterly, 31*(3). https://doi.org/10.32731/SMQ.313.0922.06

Naraine, M. L., Pegoraro, A., & Wear, H. (2021). #WeTheNorth: Examining an online brand community through a professional sport organization's hashtag marketing campaign. *Communication & Sport, 9*, 625–645. https://doi.org/10.1177/2167479519878676

Shoghanian, D., & Tang, [First Name]. (forthcoming). Narrative network centrality and club performance in the English Premier League: Evidence from press co-occurrence networks. *[Journal — in preparation]*.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x

Wanless, L., Kennedy, H., Davies, M., Naraine, M. L., & Pegoraro, A. (2024). Look what we have here: Exploring brand-related sport consumer Twitter conversation topics. *Sport Marketing Quarterly, 33*(2). https://doi.org/10.32731/SMQ.332.062024.04
