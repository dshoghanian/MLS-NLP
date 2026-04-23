"""
Collect MLS club-level xGoals and attendance data from the
American Soccer Analysis (ASA) public API.

Outputs:
  data/external/asa_xgoals.csv     — club xG, xGA, xPoints per season
  data/external/asa_attendance.csv — per-game attendance aggregated to club-season
"""
import requests
import pandas as pd
from pathlib import Path
import time

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "external"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://app.americansocceranalysis.com/api/v1/mls"
SEASONS = [str(y) for y in range(2018, 2025)]
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)"}


def get(endpoint, params=None):
    r = requests.get(f"{BASE}/{endpoint}", params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


# ── 1. Team name lookup ───────────────────────────────────────────────────────
print("Fetching team names...")
teams_raw = get("teams")
team_map = {t["team_id"]: t["team_name"] for t in teams_raw}
print(f"  {len(team_map)} teams found")

# ── 2. xGoals by club-season ─────────────────────────────────────────────────
print("Fetching xGoals data...")
xg_rows = []
for season in SEASONS:
    data = get("teams/xgoals", {"season_name": season, "split_by_seasons": "true"})
    for row in data:
        xg_rows.append({
            "club":             team_map.get(row["team_id"], row["team_id"]),
            "year":             int(season),
            "games":            row.get("count_games"),
            "goals_for":        row.get("goals_for"),
            "goals_against":    row.get("goals_against"),
            "xgoals_for":       row.get("xgoals_for"),
            "xgoals_against":   row.get("xgoals_against"),
            "xgoal_difference": row.get("xgoal_difference"),
            "gd_minus_xgd":     row.get("goal_difference_minus_xgoal_difference"),
            "points":           row.get("points"),
            "xpoints":          row.get("xpoints"),
        })
    print(f"  {season}: {len(data)} clubs")
    time.sleep(0.5)

xg_df = pd.DataFrame(xg_rows).sort_values(["year", "club"])
xg_df.to_csv(OUT_DIR / "asa_xgoals.csv", index=False)
print(f"Saved asa_xgoals.csv — {xg_df.shape}")

# ── 3. Attendance by club-season ─────────────────────────────────────────────
print("\nFetching attendance data (game-level)...")
att_rows = []
for season in SEASONS:
    games = get("games", {"season_name": season})
    # Filter regular season only (knockout_game=False) and completed games
    for g in games:
        if g.get("knockout_game") or g.get("status", "").lower() != "fulltime":
            continue
        if g.get("attendance") is None:
            continue
        att_rows.append({
            "year":         int(season),
            "home_team_id": g["home_team_id"],
            "attendance":   g["attendance"],
        })
    print(f"  {season}: {len(games)} games fetched")
    time.sleep(0.5)

att_df = pd.DataFrame(att_rows)
att_df["club"] = att_df["home_team_id"].map(team_map)

# Aggregate to club-season: mean and total home attendance
att_agg = (att_df.groupby(["club", "year"])
               .agg(
                   home_games=("attendance", "count"),
                   avg_attendance=("attendance", "mean"),
                   total_attendance=("attendance", "sum"),
               )
               .round(0)
               .reset_index())

att_agg.to_csv(OUT_DIR / "asa_attendance.csv", index=False)
print(f"Saved asa_attendance.csv — {att_agg.shape}")

# ── 4. Quick summary ─────────────────────────────────────────────────────────
print("\n=== xGoals Summary ===")
print(xg_df.describe()[["xgoals_for","xgoals_against","xpoints"]].round(2))

print("\n=== Attendance Summary ===")
print(att_agg.describe()[["avg_attendance","total_attendance"]].round(0))

print("\nDone.")
