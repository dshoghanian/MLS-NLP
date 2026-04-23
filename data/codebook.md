# Codebook of Variables
## MLS Narrative Network Analysis (2018–2024)

---

## 1. Data Sources

| Source | Type | Period | N |
|--------|------|--------|---|
| Press corpus (GDELT + RSS) | News articles | 2018–2024 | ~7,236 articles |
| Reddit fan discourse | Submissions + comments | 2018–2024 | 15,795 posts |
| MLS season standings | Official results | 2018–2024 | 28 clubs × 7 seasons |
| Forbes franchise valuations | Financial estimates | 2018–2024 | 28 clubs × 7 seasons |
| Jersey sponsor database | Sponsorship deals | 2018–2024 | 28 clubs × 7 seasons |
| ASA xGoals API | Club xG, xGA, xPoints | 2018–2024 | 186 club-season rows |
| MLSPA salary data | Player guaranteed compensation | 2018–2024 | 4,209 player records; 142 club-season rows (2020 missing) |
| ASA attendance | Home game attendance | 2018–2024 | 186 club-season rows |
| Google Trends | Weekly search interest index | 2018–2024 | 84 weekly obs × 29 clubs |

---

## 2. Unit of Analysis

The primary unit of analysis is the **club-season** (e.g., *LA Galaxy, 2022*).

Network analyses operate at the **time-window level** (annual snapshots, 2018–2024).

---

## 3. Network Construction Variables

### 3.1 Co-occurrence Network (Club ↔ Club)

| Variable | Type | Description |
|----------|------|-------------|
| `edge_weight` | Integer | Number of articles/posts in which two clubs are co-mentioned in the same document |
| `time_window` | Integer | Season year (2018–2024) |
| `network_type` | String | `club_cooccurrence` — undirected weighted graph |

### 3.2 Bipartite Network (Club ↔ Entity)

| Variable | Type | Description |
|----------|------|-------------|
| `entity_type` | String | Type of named entity: `player`, `coach`, `executive` |
| `edge_weight` | Integer | Co-mention count between club and named entity |
| `network_type` | String | `club_entity` — bipartite projection |

---

## 4. Narrative Centrality Variables

Computed per club per season from the co-occurrence network using NetworkX.

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `pagerank` | Float | 0–1 | Google PageRank score: probability a random walk lands on this club's node. Captures global narrative authority — a club mentioned alongside many other prominent clubs scores higher. |
| `pagerank_norm` | Float | 0–1 | PageRank normalized within each season (min–max) to allow cross-year comparison. |
| `degree_centrality` | Float | 0–1 | Fraction of all other clubs this club co-occurs with. High degree = broad narrative presence across many other clubs' stories. |
| `degree` | Integer | 0–27 | Raw count of unique clubs co-mentioned with this club. |
| `total_weight` | Integer | ≥0 | Sum of all edge weights incident to this node; total co-mention volume. |
| `betweenness_centrality` | Float | 0–1 | Fraction of shortest paths passing through this club's node. High betweenness = narrative bridge connecting otherwise separate storylines. |
| `closeness_centrality` | Float | 0–1 | Inverse of average shortest path to all other clubs. High closeness = rapidly reached from any point in the narrative network. |
| `eigenvector_centrality` | Float | 0–1 | Influence score weighted by neighbors' centrality. High score = mentioned alongside other highly-central clubs. |
| `narrative_rank` | Integer | 1–N | Club's rank by average PageRank within that season (1 = highest). |

---

## 5. Narrative Momentum Variables

Rolling window analysis of PageRank changes over time.

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `momentum_delta` | Float | any | Change in PageRank between consecutive windows: PageRank(T) − PageRank(T−1). |
| `momentum_label` | String | rising / falling / stable | Categorical classification of momentum_delta: `rising` (delta > +0.005), `falling` (delta < −0.005), `stable` (within ±0.005). |
| `avg_momentum` | Float | any | Mean momentum_delta across all windows in a season. |
| `windows_rising` | Integer | ≥0 | Number of monthly windows with rising momentum within the season. |
| `windows_falling` | Integer | ≥0 | Number of monthly windows with falling momentum within the season. |
| `net_momentum_signal` | Integer | any | windows_rising − windows_falling; positive = net upward narrative trajectory. |
| `narrative_trend` | String | rising / falling / stable | Season-level momentum classification based on net_momentum_signal. |
| `avg_pagerank` | Float | 0–1 | Average monthly PageRank across all windows in the season. |
| `max_pagerank` | Float | 0–1 | Peak PageRank in any single window during the season. |

---

## 6. Sentiment Variables

Computed using VADER (Valence Aware Dictionary and sEntiment Reasoner).

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `sentiment_compound` | Float | −1 to +1 | VADER compound score for the full article/post text. Values > +0.05 = positive, < −0.05 = negative, else neutral. |
| `sentiment_label` | String | positive / negative / neutral | Categorical classification of sentiment_compound. |
| `sentiment_pos` | Float | 0–1 | Proportion of text tokens with positive valence. |
| `sentiment_neg` | Float | 0–1 | Proportion of text tokens with negative valence. |
| `sentiment_neu` | Float | 0–1 | Proportion of text tokens with neutral valence. |
| `press_sent` | Float | ~0 to +0.72 | Mean VADER compound score across all press articles mentioning the club in a given year. Raw VADER range is −1 to +1; press values are predominantly positive (0 to +0.72) reflecting the generally neutral-to-positive tone of sports journalism. |
| `reddit_sent` | Float | ~−0.27 to +0.75 | Mean VADER compound score across all Reddit posts mentioning the club in a given year. Raw VADER range is −1 to +1; negative values indicate clubs with predominantly critical fan discourse (e.g., Chicago Fire FC across multiple years). |
| `sentiment_gap` | Float | any | press_sent − reddit_sent. Positive = press is more positive than fans; negative = fans are more positive than press. Both values are raw VADER compound means, making direct subtraction valid. |

---

## 7. Entity Extraction Variables

Extracted using spaCy (en_core_web_sm) combined with a domain-specific gazetteer.

| Variable | Type | Description |
|----------|------|-------------|
| `clubs_mentioned` | String (comma-delimited) | Canonical club names identified in article/post. |
| `players_mentioned` | String (comma-delimited) | Player names identified (via spaCy PERSON + roster gazetteer). |
| `coaches_mentioned` | String (comma-delimited) | Coaching staff names identified. |
| `executives_mentioned` | String (comma-delimited) | Front-office/executive names identified. |
| `club_mention_count` | Integer | Number of distinct clubs mentioned. |
| `entity_density` | Integer | Total entity mentions: clubs + players + coaches. |
| `primary_club` | String | Primary club associated with the source (for club-specific outlets/subreddits). |
| `event_types` | String (pipe-delimited) | Classified event categories present in text (e.g., `transfer`, `injury`, `match_result`). |
| `primary_event_type` | String | Dominant event type for the article/post. |

---

## 8. Temporal Context Variables

| Variable | Type | Description |
|----------|------|-------------|
| `season_phase` | String | Phase of MLS calendar: `preseason`, `regular_season`, `playoffs`, `offseason`. |
| `transfer_window` | String | Transfer market activity period: `open`, `none`. |
| `season_year` | Integer | Calendar year of publication. |
| `collection_year` | Integer | Year the article/post was collected. |
| `collection_month` | Integer | Month (1–12) the article/post was collected. |
| `published_date` | String (YYYY-MM-DD) | Publication date. |

---

## 9. Performance Variables (Ground Truth)

Sourced from official MLS season records.

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `points` | Integer | 0–102 | Total regular-season points earned (win=3, draw=1, loss=0). **Note:** 2020 values include points accumulated across the MLS is Back Tournament and shortened regular season; the 3W+D formula will not reconcile for that year due to the non-standard format. Minor discrepancies of ±1–3 points exist in other years reflecting data entry variance from official records. |
| `wins` | Integer | 0–34 | Regular-season wins. |
| `losses` | Integer | 0–34 | Regular-season losses. |
| `draws` | Integer | 0–34 | Regular-season draws. |
| `goals_for` | Integer | ≥0 | Regular-season goals scored. |
| `goals_against` | Integer | ≥0 | Regular-season goals conceded. |
| `finish` | Integer | 1–N | Final standing rank within conference. |
| `made_playoffs` | Binary | 0/1 | 1 if club qualified for MLS Cup Playoffs. |
| `conference` | String | Eastern / Western | MLS conference. |
| `cup_result` | String | categorical | Best playoff result: `MLS Cup Winner`, `MLS Cup Runner-up`, `Conference Final`, `Conference Semifinal`, `First Round`, `Missed Playoffs`, etc. |
| `performance_rank` | Integer | 1–N | Club's rank by season points (1 = most points). |
| `next_points` | Integer | 0–102 | Points earned in season T+1 (constructed lead variable for predictive regression). |

---

## 10. Market Alignment Variables

| Variable | Type | Description |
|----------|------|-------------|
| `valuation_usd_m` | Float | Estimated Forbes franchise valuation in millions USD. |
| `revenue_usd_m` | Float | Estimated total club revenue in millions USD. |
| `valuation_rank` | Integer | Club's rank by franchise valuation within season (1 = most valuable). |
| `market_gap` | Integer | valuation_rank − narrative_rank. Positive = club is "overvalued" relative to its narrative capital; negative = "undervalued" (more press attention than market value suggests). |
| `gap_category` | String | Categorical: `overexposed` (market_gap > +2), `underexposed` (market_gap < −2), `aligned` (within ±2). |

---

## 11. Brand / Sponsorship Variables

| Variable | Type | Description |
|----------|------|-------------|
| `jersey_sponsor` | String | Primary jersey kit sponsor for the season. |
| `sponsor_category` | String | Sponsor industry category (e.g., `tech`, `financial`, `automotive`). |
| `deal_value_usd_m_est` | Float | Estimated annual jersey deal value in millions USD. |
| `mention_count` | Integer | Number of articles/posts mentioning the brand keyword(s). |
| `earned_media_score` | Float | Composite score: mention_count × (1 + |sentiment_compound|). |
| `pagerank_per_dollar` | Float | Club's average PageRank divided by deal_value_usd_m_est. Proxy for narrative ROI on sponsorship investment. |

---

## 12. Reddit / Fan Discourse Variables

| Variable | Type | Description |
|----------|------|-------------|
| `subreddit` | String | Reddit community name (e.g., `r/MLS`, `r/SoundersFC`). |
| `post_type` | String | `submission` (original post) or `comment`. |
| `score` | Integer | Reddit score (upvotes − downvotes); higher = community-validated content. |
| `num_comments` | Integer | Number of replies to a submission. |
| `text_length` | Integer | Character length of post/comment text. |

### Press vs Reddit Comparative Variables

| Variable | Type | Description |
|----------|------|-------------|
| `press_rank` | Integer | Club's PageRank rank in press co-occurrence network. |
| `reddit_rank` | Integer | Club's PageRank rank in Reddit co-occurrence network. |
| `centrality_divergence` | Integer | press_rank − reddit_rank. Positive = press ranks club higher than fans do; negative = fans rank club higher than press. |
| `npi_score` | Float | Narrative Power Index: composite score of press and Reddit centrality. |
| `press_leads` | Binary | 1 if cross-correlation shows press volume peaks before Reddit volume for that club. |

---

## 13. Constructed Variables for Regression

| Variable | Formula | Description |
|----------|---------|-------------|
| `pagerank_norm` | (pagerank − min) / (max − min) per season | Season-normalized PageRank for cross-year regression |
| `momentum_rising` | 1 if momentum_label == "rising" else 0 | Indicator: club on upward narrative trajectory |
| `momentum_falling` | 1 if momentum_label == "falling" else 0 | Indicator: club on downward narrative trajectory |
| `next_points` | points shifted forward 1 year per club | Dependent variable: following season's points |

---

## 14. Sentiment Trajectory Variables

Derived from `press/sentiment_club_yearly.csv` and `reddit/sentiment_club_yearly.csv`. One row per club per season.

| Variable | Source | Type | Description |
|----------|--------|------|-------------|
| `article_count` / `post_count` | press / reddit | Integer | Number of articles (press) or posts (Reddit) mentioning the club in that season. |
| `avg_sentiment` | both | Float (−1 to +1) | Mean VADER compound score across all documents mentioning the club in the season. |
| `sentiment_pos_rate` | both | Float (0–1) | Proportion of documents with `sentiment_label == "positive"`. |
| `sentiment_neg_rate` | both | Float (0–1) | Proportion of documents with `sentiment_label == "negative"`. |
| `sentiment_neu_rate` | both | Float (0–1) | Proportion of documents with `sentiment_label == "neutral"`. |
| `sentiment_std` | both | Float | Standard deviation of VADER compound scores; high values indicate polarised coverage. |
| `avg_score` | reddit only | Float | Mean Reddit upvote score for posts mentioning the club. Higher = community-validated content. |

---

## 15. External Performance & Market Variables

Sourced from American Soccer Analysis API and MLSPA salary disclosures. One row per club per season. Joined to master panel by `club` + `year`.

### 15.1 ASA xGoals (`asa_xgoals.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `games` | Integer | Number of regular-season games played. |
| `goals_for` | Integer | Regular-season goals scored. |
| `goals_against` | Integer | Regular-season goals conceded. |
| `xgoals_for` | Float | Expected goals created (shot quality × location). |
| `xgoals_against` | Float | Expected goals conceded. |
| `xgoal_difference` | Float | `xgoals_for − xgoals_against`. Positive = generated better chances than conceded. |
| `gd_minus_xgd` | Float | Actual goal difference minus expected goal difference. Positive = outperformed expected (luck or clinical finishing). |
| `xpoints` | Float | Points expected based on shot quality. Closer to `points` = consistent finishing; large divergence = luck or clutch factor. |

### 15.2 ASA Attendance (`asa_attendance.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `home_games` | Integer | Number of regular-season home games with recorded attendance. |
| `avg_attendance` | Float | Mean home game attendance. |
| `total_attendance` | Integer | Sum of home game attendance across the season. |

### 15.3 MLSPA Salaries (`mlspa_salaries.csv`, `mlspa_players.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `player_count` | Integer | Number of players on salary disclosure for the club-season. |
| `total_payroll` | Float | Sum of guaranteed compensation across all disclosed players (USD). |
| `avg_salary` | Float | Mean guaranteed compensation per player. |
| `median_salary` | Float | Median guaranteed compensation; less sensitive to DP outliers than mean. |
| `max_salary` | Float | Highest single-player guaranteed compensation; proxy for DP investment. |
| `base_salary` | Float | Player-level base salary (player file only). |
| `guaranteed_comp` | Float | Player-level guaranteed compensation including bonuses (player file only). |

**Note:** 2020 data is missing from MLSPA disclosures (COVID-shortened season non-standard reporting). Salary data covers 2018–2019 and 2021–2024.

### 15.4 Google Trends (`google_trends.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `avg_search_interest` | Float | Mean weekly Google Trends search interest index (0–100) for the club name, averaged across the calendar year. Index is relative; cross-batch normalization applied via anchor term "MLS soccer." |
| `max_search_interest` | Float | Peak weekly search interest in the year. |
| `weeks_above_50` | Integer | Number of weeks where search interest exceeded 50 (half the maximum). Captures sustained public attention vs. brief spikes. |

### 15.5 Derived External Variables (Extended Regression)

| Variable | Formula | Description |
|----------|---------|-------------|
| `log_payroll` | `log(total_payroll + 1)` | Log-transformed payroll for normality; reduces leverage of extreme DP contracts. |
| `log_attendance` | `log(avg_attendance + 1)` | Log-transformed home attendance. |
| `search_norm` | Season min-max normalized `avg_search_interest` | Google Trends index normalized within each season for cross-year comparability. |

---

*Codebook version 1.2 — MLS NLP Project, 2018–2024 (updated April 2026: external data sources added — ASA xGoals, ASA attendance, MLSPA salaries, Google Trends; topic labels updated after LDA retraining with domain stopwords)*
