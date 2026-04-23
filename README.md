# MLS Narrative Network Analysis (2018–2024)

A network analysis and NLP study examining the relationship between **narrative capital** — a club's prominence in press and fan discourse — and competitive performance across 28 Major League Soccer clubs from 2018 to 2024.

---

## What This Project Does

We build co-occurrence networks from two text corpora:
- **~7,200 press articles** (GDELT + club RSS feeds)
- **~15,800 Reddit posts/comments** (r/MLS + 28 club subreddits)

For each year, clubs that appear together in the same article or post are connected by an edge. We run **Google PageRank** on these networks to measure each club's *narrative centrality* — not just how often they're mentioned, but how central they are to the overall media conversation.

We then ask: **does being talked about more actually help you win?**

---

## Key Findings

| Finding | Result |
|---------|--------|
| Narrative rank ~ Performance rank | r = 0.41 |
| Average narrative-performance gap | 6.6 rank positions |
| Does narrative predict next-season points? | No — F = 0.72, p = 0.578 |
| LOYO cross-validation median R² | −0.07 (doesn't generalize) |
| Only significant predictor of future points | `made_playoffs` (β = +7.2, p < 0.001) |
| Messi signing causal effect (SCM) | +0.035 PageRank units above counterfactual |
| Press vs. Reddit sentiment gap | Press consistently +0.18 more positive |

**The main finding:** narrative centrality and competitive performance are largely decoupled. Clubs can hold outsized narrative attention without winning, and win consistently without media dominance. This makes narrative capital a distinct organizational resource — not just a reflection of on-field results.

---

## Project Structure

```
MLS_NLP/
├── notebooks/
│   ├── 00_showcase.ipynb          # Main analysis notebook — runs everything end to end
│   ├── 01_data_collection.ipynb
│   ├── 02_nlp_enrichment.ipynb
│   ├── 03_network_construction.ipynb
│   ├── 04_centrality_analysis.ipynb
│   ├── 05_market_alignment.ipynb
│   ├── 06_reddit_fan_discourse.ipynb
│   └── 07_predictive_regression.ipynb
│
├── scripts/
│   ├── collect_trends.py          # Google Trends data collection (pytrends)
│   └── topic_modeling.py          # LDA topic model training (Gensim)
│
├── pipeline/                      # Data collection and enrichment pipeline
│
├── data/
│   ├── codebook.md                # Full variable definitions (v1.2)
│   ├── analysis/                  # Output CSVs, plots, reports
│   │   ├── press/                 # Press network centrality, sentiment, brand analysis
│   │   ├── reddit/                # Reddit network centrality and sentiment
│   │   ├── shared/                # Topic model outputs, master panel, regression results
│   │   └── comparison/            # Press vs. Reddit comparative analysis
│   ├── external/                  # ASA xGoals, MLSPA salaries, attendance, Google Trends
│   ├── performance/               # MLS standings, Forbes valuations, sponsorship data
│   └── press/networks/            # Quarterly co-occurrence network files (parquet + JSON)
│       └── reddit/networks/       # Monthly co-occurrence network files
│
├── config/                        # Entity lists, settings, data sources
├── app.py                         # Interactive Streamlit network explorer
└── requirements.txt
```

> **Note:** `data/press/raw/` and `data/reddit/raw/` (raw article/post text) are excluded from this repo due to size (~2.1 GB). All derived outputs (networks, centrality scores, sentiment, topics) are included.

---

## Data Sources

| Source | N | Purpose |
|--------|---|---------|
| Press corpus (GDELT + RSS) | ~7,236 articles | Network construction, sentiment, topics |
| Reddit fan discourse | 15,795 posts | Fan network, sentiment, topics |
| MLS season standings | 186 club-seasons | Performance ground truth |
| Forbes franchise valuations | 186 club-seasons | Market size control |
| Jersey sponsor database | 186 club-seasons | Sponsorship ROI analysis |
| ASA xGoals API | 186 club-seasons | Process performance control |
| MLSPA salary data | 142 club-seasons | Payroll control (2020 missing) |
| ASA attendance | 186 club-seasons | Fan demand control |
| Google Trends | ~84 weekly obs × 28 clubs | Public interest index |

---

## Methods

- **Network construction:** spaCy NER + domain gazetteer for entity extraction; NetworkX for graph construction and centrality computation (PageRank, degree, betweenness, closeness, eigenvector)
- **Sentiment analysis:** VADER (Hutto & Gilbert, 2014) on all press and Reddit text
- **Topic modeling:** LDA via Gensim (7 topics), retrained with domain-specific stopwords to remove non-MLS sports contamination
- **Regression:** OLS (3 nested models), Two-Way Fixed Effects (linearmodels, club + year FE, clustered SE), Logistic regression, Leave-One-Year-Out cross-validation, extended models with xGoals/payroll/attendance controls
- **Causal inference:** Difference-in-Differences + Synthetic Control Method (pysyncon) for the Messi natural experiment
- **Temporal analysis:** Granger causality on 27-quarter aggregate series reconstructed from quarterly network parquets; cross-correlation lag analysis

---

## Running the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main showcase notebook
jupyter lab notebooks/00_showcase.ipynb

# Launch the interactive Streamlit app
streamlit run app.py
```

---

## Codebook

Full variable definitions are in [`data/codebook.md`](data/codebook.md). Covers 15 sections including network construction variables, narrative centrality, momentum, sentiment, entity extraction, performance, market alignment, brand/sponsorship, and all external data sources.


