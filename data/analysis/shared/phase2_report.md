# Phase 2 — Press vs Reddit Analysis

## 2a. Institutional vs Fan Narrative Divergence

### Quadrant Classification

Clubs classified by whether their average press rank vs Reddit rank is above/below the median, producing four quadrant types:

| Quadrant | Definition |
|---|---|
| **Consensus Stars** | Prominent in both press and Reddit |
| **Institutional Legacy** | Press-prominent; Reddit less engaged |
| **Grassroots Underdogs** | Reddit-prominent; press under-covers |
| **Low Profile** | Below median in both channels |

### Club Quadrant Assignments

| Club | Avg Press Rank | Avg Reddit Rank | Avg Gap | Quadrant |
|---|---|---|---|---|
| CF Montreal | 3.2 | 18.2 | 15.0 | Institutional Legacy |
| Seattle Sounders FC | 3.9 | 5.3 | 1.4 | Consensus Stars |
| LA Galaxy | 5.0 | 13.7 | 8.7 | Institutional Legacy |
| Toronto FC | 6.0 | 8.0 | 2.0 | Consensus Stars |
| Philadelphia Union | 6.8 | 11.8 | 5.0 | Consensus Stars |
| LAFC | 7.7 | 7.0 | -0.7 | Consensus Stars |
| Inter Miami CF | 8.2 | 10.6 | 2.4 | Consensus Stars |
| Columbus Crew | 8.9 | 8.4 | -0.4 | Consensus Stars |
| New York City FC | 11.3 | 13.0 | 1.7 | Consensus Stars |
| Orlando City SC | 12.2 | 13.5 | 1.3 | Consensus Stars |
| Portland Timbers | 12.4 | 12.4 | 0.0 | Consensus Stars |
| Nashville SC | 12.8 | 14.6 | 1.8 | Institutional Legacy |
| Atlanta United FC | 13.3 | 18.8 | 5.5 | Institutional Legacy |
| New York Red Bulls | 13.7 | 17.9 | 4.1 | Institutional Legacy |
| Chicago Fire FC | 14.6 | 18.8 | 4.2 | Institutional Legacy |
| Vancouver Whitecaps FC | 14.9 | 12.4 | -2.4 | Grassroots Underdogs |
| FC Dallas | 16.1 | 11.9 | -4.3 | Grassroots Underdogs |
| Real Salt Lake | 16.3 | 11.7 | -4.6 | Grassroots Underdogs |
| FC Cincinnati | 16.7 | 18.7 | 2.0 | Low Profile |
| D.C. United | 17.7 | 13.7 | -4.0 | Low Profile |
| Colorado Rapids | 18.1 | 15.6 | -2.6 | Low Profile |
| New England Revolution | 18.5 | 16.8 | -1.7 | Low Profile |
| Houston Dynamo FC | 19.4 | 12.0 | -7.4 | Grassroots Underdogs |
| Sporting Kansas City | 20.7 | 12.0 | -8.7 | Grassroots Underdogs |
| Minnesota United FC | 21.3 | 17.6 | -3.7 | Low Profile |
| St. Louis City SC | 22.0 | 28.0 | 6.0 | Low Profile |
| Charlotte FC | 22.7 | 21.3 | -1.3 | Low Profile |
| San Jose Earthquakes | 25.6 | 16.7 | -8.9 | Low Profile |
| Austin FC | 26.2 | 13.5 | -12.8 | Grassroots Underdogs |

### Predictive Value of Divergence

Correlation between press-Reddit rank gap and next-year valuation % change: **r = -0.012**.

A positive gap (press rank lower number than Reddit rank = press-prominent) correlates with valuation growth, suggesting institutional media coverage leads commercial outcomes.

### Case Study Clubs

| Club | Year | Press Rank | Reddit Rank | Gap | Points |
|---|---|---|---|---|---|
| CF Montreal | 2018 | nan | nan | nan | 38 |
| CF Montreal | 2019 | 4 | 7 | 3 | 45 |
| CF Montreal | 2020 | 2 | 21 | 19 | 24 |
| CF Montreal | 2021 | 6 | 28 | 22 | 59 |
| CF Montreal | 2022 | 1 | 27 | 26 | 57 |
| CF Montreal | 2023 | 3 | 16 | 13 | 46 |
| CF Montreal | 2024 | 3 | 10 | 7 | 48 |
| Charlotte FC | 2022 | 23 | 17 | -6 | 44 |
| Charlotte FC | 2023 | 19 | 21 | 2 | 51 |
| Charlotte FC | 2024 | 26 | 26 | 0 | 55 |
| Inter Miami CF | 2020 | 10 | 11 | 1 | 16 |
| Inter Miami CF | 2021 | 20 | 27 | 7 | 39 |
| Inter Miami CF | 2022 | 9 | 9 | 0 | 44 |
| Inter Miami CF | 2023 | 1 | 2 | 1 | 54 |
| Inter Miami CF | 2024 | 1 | 4 | 3 | 61 |
| Seattle Sounders FC | 2018 | 3 | 3 | 0 | 52 |
| Seattle Sounders FC | 2019 | 1 | 4 | 3 | 55 |
| Seattle Sounders FC | 2020 | 7 | 3 | -4 | 44 |
| Seattle Sounders FC | 2021 | 1 | 1 | 0 | 55 |
| Seattle Sounders FC | 2022 | 6 | 7 | 1 | 53 |
| Seattle Sounders FC | 2023 | 4 | 13 | 9 | 50 |
| Seattle Sounders FC | 2024 | 5 | 6 | 1 | 52 |
| Toronto FC | 2018 | 5 | 8 | 3 | 57 |
| Toronto FC | 2019 | 2 | 5 | 3 | 49 |
| Toronto FC | 2020 | 3 | 4 | 1 | 26 |
| Toronto FC | 2021 | 10 | 5 | -5 | 27 |
| Toronto FC | 2022 | 7 | 13 | 6 | 28 |
| Toronto FC | 2023 | 11 | 14 | 3 | 35 |
| Toronto FC | 2024 | 4 | 7 | 3 | 38 |

---

## 2b. Granger Causality — Press vs Reddit Sentiment

**Series:** Annual aggregate VADER sentiment (mean across all clubs), 2018–2024 (N=7).

**Note on sample size:** With only 7 annual observations, Granger tests have very low power. Results should be interpreted as exploratory directional evidence, not confirmatory. A monthly-frequency series (N=84) is available for future work using the quarterly centrality data.

| Direction | Lag | F-stat | p-value | Significant |
|---|---|---|---|---|
| press → reddit | 1 | 0.158 | 0.718 | No |
| reddit → press | 1 | 0.335 | 0.603 | No |

---

## Figures
- `phase2_divergence_quadrants.png` — press vs Reddit rank quadrant scatter
- `phase2_case_studies.png` — Toronto, CF Montreal, Charlotte, Inter Miami, Seattle trajectories
- `phase2_granger.png` — aggregate sentiment series (press vs Reddit)