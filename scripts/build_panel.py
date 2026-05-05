"""
Build the MLS master analysis panel — equivalent to EPL's club_analysis_panel.

Merges: master_summary, cooccurrence centrality, narrative summaries,
press/reddit comparison, sentiment, salaries, attendance, xgoals,
valuations, brand/sponsor data.

Adds EPL-equivalent normalization: strength_norm, strength_z,
comp_share, in_giant, press_to_reddit_ratio.

Output: data/analysis/shared/mls_analysis_panel.csv
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Name normalisation map (external files → master_summary naming)
# ---------------------------------------------------------------------------
CLUB_RENAME = {
    "CF Montréal": "CF Montreal",
    "Los Angeles FC": "LAFC",
    "Portland Timbers FC": "Portland Timbers",
}


def norm_name(series):
    return series.replace(CLUB_RENAME)


# ---------------------------------------------------------------------------
# 1. Base: master_summary
# ---------------------------------------------------------------------------
base = pd.read_csv("data/analysis/shared/master_summary.csv")
# base columns: entity, year, narrative_rank, performance_rank, gap,
#               pagerank, degree, total_weight, momentum_label,
#               points, wins, losses, draws, finish, made_playoffs,
#               cup_result, conference
panel = base.copy()
N_BASE = len(panel)

# ---------------------------------------------------------------------------
# 2. Cooccurrence network — yearly windows only, clubs only
#    Adds: press_cooc_pagerank, press_cooc_degree, press_cooc_total_weight,
#          press_cooc_eigenvector, press_cooc_betweenness, press_cooc_closeness
#          press_strength_norm, press_strength_z, press_comp_share, press_in_giant
# ---------------------------------------------------------------------------
cooc_raw = pd.read_csv("data/analysis/press/centrality_club_cooccurrence.csv")
cooc = cooc_raw[
    (cooc_raw["entity_type"] == "club")
    & (cooc_raw["time_window"].astype(str).str.match(r"^\d{4}$"))
].copy()
cooc["year"] = cooc["time_window"].astype(int)

# Within-year normalization
cooc["press_strength_norm"] = cooc.groupby("year")["total_weight"].transform(
    lambda x: x / x.sum()
)
cooc["press_strength_z"] = cooc.groupby("year")["total_weight"].transform(
    lambda x: (x - x.mean()) / x.std()
)

cooc_cols = {
    "entity": "entity",
    "year": "year",
    "pagerank": "press_cooc_pagerank",
    "degree": "press_cooc_degree",
    "total_weight": "press_cooc_strength",
    "eigenvector_centrality": "press_cooc_eigenvector",
    "betweenness_centrality": "press_cooc_betweenness",
    "closeness_centrality": "press_cooc_closeness",
    "press_strength_norm": "press_strength_norm",
    "press_strength_z": "press_strength_z",
}
cooc_merge = cooc.rename(columns=cooc_cols)[list(cooc_cols.values())]
panel = panel.merge(cooc_merge, on=["entity", "year"], how="left")

# ---------------------------------------------------------------------------
# 3. Narrative summary — cooccurrence (aggregated quarterly momentum)
#    Adds: press_windows_rising, press_windows_falling, press_net_momentum
# ---------------------------------------------------------------------------
ns_cooc = pd.read_csv("data/analysis/press/narrative_summary_club_cooccurrence.csv")
ns_cooc_cols = {
    "entity": "entity",
    "year": "year",
    "avg_pagerank": "press_avg_pagerank",
    "max_pagerank": "press_max_pagerank",
    "windows_rising": "press_windows_rising",
    "windows_falling": "press_windows_falling",
    "net_momentum_signal": "press_net_momentum",
    "narrative_trend": "press_trend",
}
panel = panel.merge(
    ns_cooc.rename(columns=ns_cooc_cols)[list(ns_cooc_cols.values())],
    on=["entity", "year"],
    how="left",
)

# ---------------------------------------------------------------------------
# 4. Press vs Reddit centrality comparison
#    Adds: press_rank, reddit_rank, press_reddit_rank_gap,
#          press_pagerank (comparison), reddit_pagerank (comparison)
# ---------------------------------------------------------------------------
pvr = pd.read_csv("data/analysis/comparison/press_vs_reddit_centrality.csv")
pvr_cols = {
    "entity": "entity",
    "year": "year",
    "press_pagerank": "press_pagerank",
    "press_degree": "press_degree",
    "press_rank": "press_rank",
    "reddit_pagerank": "reddit_pagerank",
    "reddit_degree": "reddit_degree",
    "reddit_rank": "reddit_rank",
    "rank_gap": "press_reddit_rank_gap",
}
panel = panel.merge(
    pvr.rename(columns=pvr_cols)[list(pvr_cols.values())],
    on=["entity", "year"],
    how="left",
)

# Compute press-to-reddit ratio
panel["press_to_reddit_ratio"] = panel["press_pagerank"] / panel["reddit_pagerank"].replace(0, np.nan)

# Within-year z-scores using the comparison pageranks
panel["press_pagerank_z"] = panel.groupby("year")["press_pagerank"].transform(
    lambda x: (x - x.mean()) / x.std()
)
panel["reddit_pagerank_z"] = panel.groupby("year")["reddit_pagerank"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# ---------------------------------------------------------------------------
# 5. Press sentiment (single-club rows only)
#    Adds: press_article_count, press_avg_sentiment, press_sentiment_pos/neg rates
# ---------------------------------------------------------------------------
ps = pd.read_csv("data/analysis/press/sentiment_club_yearly.csv")
ps = ps[~ps["club"].str.contains("|", regex=False)].copy()
ps = ps.rename(columns={
    "club": "entity",
    "season_year": "year",
    "article_count": "press_article_count",
    "avg_sentiment": "press_avg_sentiment",
    "sentiment_pos_rate": "press_sentiment_pos_rate",
    "sentiment_neg_rate": "press_sentiment_neg_rate",
    "sentiment_neu_rate": "press_sentiment_neu_rate",
    "sentiment_std": "press_sentiment_std",
})
panel = panel.merge(
    ps[["entity", "year", "press_article_count", "press_avg_sentiment",
        "press_sentiment_pos_rate", "press_sentiment_neg_rate",
        "press_sentiment_neu_rate", "press_sentiment_std"]],
    on=["entity", "year"],
    how="left",
)

# ---------------------------------------------------------------------------
# 6. Reddit narrative summary
#    Adds: reddit_avg_pagerank, reddit_max_pagerank, reddit_windows_rising/falling
# ---------------------------------------------------------------------------
rn = pd.read_csv("data/analysis/reddit/narrative_summary_club_reddit.csv")
rn_cols = {
    "entity": "entity",
    "year": "year",
    "avg_pagerank": "reddit_avg_pagerank",
    "max_pagerank": "reddit_max_pagerank",
    "windows_rising": "reddit_windows_rising",
    "windows_falling": "reddit_windows_falling",
    "net_momentum_signal": "reddit_net_momentum",
    "narrative_trend": "reddit_trend",
}
panel = panel.merge(
    rn.rename(columns=rn_cols)[list(rn_cols.values())],
    on=["entity", "year"],
    how="left",
)

# Within-year z-scores for reddit
panel["reddit_avg_pagerank_z"] = panel.groupby("year")["reddit_avg_pagerank"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# ---------------------------------------------------------------------------
# 7. Reddit sentiment
# ---------------------------------------------------------------------------
rs = pd.read_csv("data/analysis/reddit/sentiment_club_yearly.csv")
rs = rs[~rs["club"].str.contains("|", regex=False)].copy()
rs = rs.rename(columns={
    "club": "entity",
    "season_year": "year",
    "post_count": "reddit_post_count",
    "avg_sentiment": "reddit_avg_sentiment",
    "sentiment_pos_rate": "reddit_sentiment_pos_rate",
    "sentiment_neg_rate": "reddit_sentiment_neg_rate",
    "avg_score": "reddit_avg_score",
})
panel = panel.merge(
    rs[["entity", "year", "reddit_post_count", "reddit_avg_sentiment",
        "reddit_sentiment_pos_rate", "reddit_sentiment_neg_rate", "reddit_avg_score"]],
    on=["entity", "year"],
    how="left",
)

# Sentiment gap (press minus reddit)
panel["sentiment_gap"] = panel["press_avg_sentiment"] - panel["reddit_avg_sentiment"]

# ---------------------------------------------------------------------------
# 8. Salaries (MLSPA) — missing 2020
# ---------------------------------------------------------------------------
sal = pd.read_csv("data/external/mlspa_salaries.csv")
sal = sal.rename(columns={"club": "entity"})
panel = panel.merge(
    sal[["entity", "year", "player_count", "total_payroll", "avg_salary",
         "median_salary", "max_salary"]],
    on=["entity", "year"],
    how="left",
)

# ---------------------------------------------------------------------------
# 9. Attendance — fix name mismatches
# ---------------------------------------------------------------------------
att = pd.read_csv("data/external/asa_attendance.csv")
att["club"] = norm_name(att["club"])
att = att.rename(columns={"club": "entity"})
panel = panel.merge(
    att[["entity", "year", "avg_attendance", "total_attendance"]],
    on=["entity", "year"],
    how="left",
)

# ---------------------------------------------------------------------------
# 10. xGoals — fix name mismatches
# ---------------------------------------------------------------------------
xg = pd.read_csv("data/external/asa_xgoals.csv")
xg["club"] = norm_name(xg["club"])
xg = xg.rename(columns={"club": "entity"})
panel = panel.merge(
    xg[["entity", "year", "xgoals_for", "xgoals_against", "xgoal_difference",
        "gd_minus_xgd", "xpoints"]],
    on=["entity", "year"],
    how="left",
)

# ---------------------------------------------------------------------------
# 11. Valuations — FC Cincinnati missing
# ---------------------------------------------------------------------------
val = pd.read_csv("data/performance/mls_valuations.csv")
val = val.rename(columns={"club": "entity"})
panel = panel.merge(
    val[["entity", "year", "valuation_usd_m", "revenue_usd_m"]],
    on=["entity", "year"],
    how="left",
)

# Within-year valuation z-score
panel["valuation_z"] = panel.groupby("year")["valuation_usd_m"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# ---------------------------------------------------------------------------
# 12. Brand / sponsor
# ---------------------------------------------------------------------------
bd = pd.read_csv("data/analysis/press/brand_deal_value.csv")
bd = bd.rename(columns={"entity": "entity"})  # already named correctly
# brand_deal has both 'club' and 'entity' columns — use 'entity' as the key
panel = panel.merge(
    bd[["entity", "year", "jersey_sponsor", "sponsor_category",
        "deal_value_usd_m_est", "pagerank_per_dollar", "sponsor_value_score"]],
    on=["entity", "year"],
    how="left",
)

# ---------------------------------------------------------------------------
# 13. Final derived columns
# ---------------------------------------------------------------------------

# Payroll per point (efficiency)
panel["payroll_per_point"] = panel["total_payroll"] / panel["points"].replace(0, np.nan)

# Narrative efficiency: pagerank per payroll dollar (millions)
panel["narrative_per_payroll"] = panel["press_cooc_pagerank"] / (
    panel["total_payroll"] / 1_000_000
).replace(0, np.nan)

# xPoints gap: actual points vs xpoints (luck / over/underperformance)
panel["points_vs_xpoints"] = panel["points"] - panel["xpoints"]

# ---------------------------------------------------------------------------
# 14. Validate and save
# ---------------------------------------------------------------------------
assert len(panel) == N_BASE, f"Row count changed: {N_BASE} → {len(panel)}"

out_path = "data/analysis/shared/mls_analysis_panel.csv"
panel.to_csv(out_path, index=False)

print(f"Panel saved: {out_path}")
print(f"Shape: {panel.shape}")
print(f"\nColumns ({len(panel.columns)}):")
for c in panel.columns:
    n_null = panel[c].isna().sum()
    pct = 100 * n_null / len(panel)
    print(f"  {c:<45} nulls: {n_null:3d} ({pct:4.1f}%)")
