"""
Collect Google Trends weekly search interest for all MLS clubs (2018-2024).

Google Trends returns relative interest (0-100) within each query batch.
To make clubs comparable across batches, we use a common anchor term
("MLS soccer") present in every batch, then rescale all club series
to the anchor's baseline.

Output:
  data/external/google_trends_raw.csv   — raw batch values
  data/external/google_trends.csv       — normalized, club-year aggregated
"""
import time
import pandas as pd
from pathlib import Path
from pytrends.request import TrendReq

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "external"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME = "2018-01-01 2024-12-31"
GEO       = "US"
CAT       = 20   # Sports category

ANCHOR = "MLS soccer"   # Present in every batch for cross-batch normalization

# All 28 clubs in our dataset
CLUBS = [
    "Atlanta United FC", "Austin FC", "CF Montreal", "Charlotte FC",
    "Chicago Fire FC", "Colorado Rapids", "Columbus Crew", "D.C. United",
    "FC Cincinnati", "FC Dallas", "Houston Dynamo FC", "Inter Miami CF",
    "LA Galaxy", "LAFC", "Minnesota United FC", "Nashville SC",
    "New England Revolution", "New York City FC", "New York Red Bulls",
    "Orlando City SC", "Philadelphia Union", "Portland Timbers",
    "Real Salt Lake", "San Jose Earthquakes", "Seattle Sounders FC",
    "Sporting Kansas City", "St. Louis City SC", "Toronto FC",
    "Vancouver Whitecaps FC",
]

# Google Trends allows max 5 terms per request; anchor uses one slot → 4 clubs/batch
BATCH_SIZE = 4
SLEEP_SEC  = 8   # Be polite; avoid rate-limiting


def build_batches(clubs, batch_size=BATCH_SIZE):
    batches = []
    for i in range(0, len(clubs), batch_size):
        batches.append(clubs[i:i + batch_size])
    return batches


pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=3, backoff_factor=1)

print(f"Collecting Google Trends for {len(CLUBS)} clubs in batches of {BATCH_SIZE}...")
print(f"Timeframe: {TIMEFRAME} | Geo: {GEO} | Category: {CAT} (Sports)")
print(f"Anchor term: '{ANCHOR}'\n")

# ── 1. Get anchor baseline ────────────────────────────────────────────────────
pt.build_payload([ANCHOR], cat=CAT, timeframe=TIMEFRAME, geo=GEO)
anchor_df = pt.interest_over_time()
if anchor_df.empty or ANCHOR not in anchor_df.columns:
    raise RuntimeError("Failed to fetch anchor baseline from Google Trends.")
anchor_series = anchor_df[ANCHOR]
anchor_mean   = anchor_series.mean()
print(f"Anchor mean interest: {anchor_mean:.2f}")

# ── 2. Fetch each batch ───────────────────────────────────────────────────────
all_raw = {}
batches  = build_batches(CLUBS)

for i, batch in enumerate(batches):
    kws = [ANCHOR] + batch
    print(f"Batch {i+1}/{len(batches)}: {batch}")
    try:
        pt.build_payload(kws, cat=CAT, timeframe=TIMEFRAME, geo=GEO)
        df = pt.interest_over_time()
        if df.empty:
            print("  [empty response — skipping]")
            time.sleep(SLEEP_SEC * 2)
            continue
        # Compute scale factor: how does this batch's anchor compare to baseline?
        batch_anchor_mean = df[ANCHOR].mean()
        scale = anchor_mean / batch_anchor_mean if batch_anchor_mean > 0 else 1.0
        for club in batch:
            if club in df.columns:
                # Apply scale factor to normalize across batches
                all_raw[club] = (df[club] * scale).clip(upper=100)
        print(f"  scale={scale:.3f} | anchor_mean={batch_anchor_mean:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(SLEEP_SEC)

# ── 3. Combine into wide DataFrame ───────────────────────────────────────────
trends_df = pd.DataFrame(all_raw)
trends_df.index.name = "date"
trends_df = trends_df.sort_index()

# Drop 'isPartial' if somehow included
trends_df = trends_df[[c for c in trends_df.columns if c in CLUBS]]

raw_path = OUT_DIR / "google_trends_raw.csv"
trends_df.to_csv(raw_path)
print(f"\nSaved google_trends_raw.csv — {trends_df.shape}")

# ── 4. Aggregate to club-year ─────────────────────────────────────────────────
trends_df.index = pd.to_datetime(trends_df.index)
trends_long = trends_df.stack().reset_index()
trends_long.columns = ["date", "club", "search_interest"]
trends_long["year"] = trends_long["date"].dt.year

# Keep only 2018-2024
trends_long = trends_long[trends_long["year"].between(2018, 2024)]

agg = (trends_long.groupby(["club", "year"])
                  .agg(
                      avg_search_interest=("search_interest", "mean"),
                      max_search_interest=("search_interest", "max"),
                      weeks_above_50=("search_interest", lambda x: (x > 50).sum()),
                  )
                  .round(2)
                  .reset_index())

out_path = OUT_DIR / "google_trends.csv"
agg.to_csv(out_path, index=False)
print(f"Saved google_trends.csv — {agg.shape}")

print("\n=== Top clubs by average search interest (2018-2024) ===")
top = (agg.groupby("club")["avg_search_interest"]
          .mean()
          .sort_values(ascending=False)
          .head(10))
print(top.round(2).to_string())

print("\nDone.")
