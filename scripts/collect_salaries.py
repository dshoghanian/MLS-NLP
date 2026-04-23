"""
Collect MLSPA player salary data (2018-2024) and aggregate to club-season level.

Sources:
  2024: CSV directly from MLSPA S3
  2018-2023: PDFs from MLSPA S3, parsed with pdfplumber

Output:
  data/external/mlspa_salaries.csv  — club-season aggregated payroll
  data/external/mlspa_players.csv   — player-level salary data (all years)
"""
import re
import io
import time
import requests
import pdfplumber
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "external"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)"}

SOURCES = {
    2024: ("csv",      "http://s3.amazonaws.com/mlspa/Salary-Release-FALL-2024_241024_164547.csv"),
    2023: ("pdf_tbl",  "http://s3.amazonaws.com/mlspa/2023-Salary-Report-as-of-Sept-15-2023.pdf"),
    2022: ("pdf_tbl",  "http://s3.amazonaws.com/mlspa/2022-Fall-Salary-Guide.pdf"),
    2021: ("pdf_tbl",  "http://s3.amazonaws.com/mlspa/2021-MLSPA-Fall-Salary-release.pdf"),
    2020: ("pdf_txt",  "http://s3.amazonaws.com/mlspa/2020-Fall-Winter-Salary-List-alphabetical.pdf"),
    2019: ("pdf_txt",  "http://s3.amazonaws.com/mlspa/Salary-List-Fall-Release-FINAL-Salary-List-Fall-Release-MLS.pdf"),
    2018: ("pdf_txt",  "http://s3.amazonaws.com/mlspa/2018-09-15-Salary-Information-Alphabetical.pdf"),
}

# Club name normalisation — map MLSPA names → canonical project names
CLUB_MAP = {
    "Atlanta United":           "Atlanta United FC",
    "Austin FC":                "Austin FC",
    "CF Montréal":              "CF Montreal",
    "CF Montreal":              "CF Montreal",
    "Charlotte FC":             "Charlotte FC",
    "Chicago Fire":             "Chicago Fire FC",
    "Chicago Fire FC":          "Chicago Fire FC",
    "FC Cincinnati":            "FC Cincinnati",
    "Colorado Rapids":          "Colorado Rapids",
    "Columbus Crew":            "Columbus Crew",
    "Columbus Crew SC":         "Columbus Crew",
    "D.C. United":              "D.C. United",
    "DC United":                "D.C. United",
    "FC Dallas":                "FC Dallas",
    "Houston Dynamo":           "Houston Dynamo FC",
    "Houston Dynamo FC":        "Houston Dynamo FC",
    "Inter Miami CF":           "Inter Miami CF",
    "Inter Miami":              "Inter Miami CF",
    "LA Galaxy":                "LA Galaxy",
    "Los Angeles FC":           "LAFC",
    "LAFC":                     "LAFC",
    "Minnesota United":         "Minnesota United FC",
    "Minnesota United FC":      "Minnesota United FC",
    "Nashville SC":             "Nashville SC",
    "New England Revolution":   "New England Revolution",
    "New York City FC":         "New York City FC",
    "New York Red Bulls":       "New York Red Bulls",
    "NYCFC":                    "New York City FC",
    "Orlando City":             "Orlando City SC",
    "Orlando City SC":          "Orlando City SC",
    "Philadelphia Union":       "Philadelphia Union",
    "Portland Timbers":         "Portland Timbers",
    "Real Salt Lake":           "Real Salt Lake",
    "San Jose Earthquakes":     "San Jose Earthquakes",
    "Seattle Sounders":         "Seattle Sounders FC",
    "Seattle Sounders FC":      "Seattle Sounders FC",
    "Sporting Kansas City":     "Sporting Kansas City",
    "SKC":                      "Sporting Kansas City",
    "St. Louis City SC":        "St. Louis City SC",
    "Toronto FC":               "Toronto FC",
    "Vancouver Whitecaps":      "Vancouver Whitecaps FC",
    "Vancouver Whitecaps FC":   "Vancouver Whitecaps FC",
    "St Louis City SC":         "St. Louis City SC",
}


def normalize_club(name):
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    return CLUB_MAP.get(name, name)


def parse_salary_str(s):
    """Convert '$1,234,567.00' or '$ 6 8,927.00' → float"""
    if not isinstance(s, str):
        return None
    # Remove $, spaces within digits, commas
    s = s.replace("$", "").replace(",", "").strip()
    # Remove internal spaces (artifact of PDF text extraction)
    s = re.sub(r"(\d)\s+(\d)", r"\1\2", s).strip()
    try:
        return float(s)
    except ValueError:
        return None


# ── 1. CSV years ──────────────────────────────────────────────────────────────
def load_csv(year, url):
    print(f"  {year}: downloading CSV...")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Standardize columns
    df.columns = [c.strip() for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        club = normalize_club(str(row.get("club", "")))
        base = parse_salary_str(str(row.get("CY Base Salary", "")))
        comp = parse_salary_str(str(row.get("CY Guaranteed Comp", "")))
        if club and comp:
            rows.append({"year": year, "club": club,
                         "base_salary": base, "guaranteed_comp": comp})
    print(f"    {len(rows)} player records")
    return pd.DataFrame(rows)


# ── 2. PDF years ──────────────────────────────────────────────────────────────
SALARY_RE  = re.compile(r"\$[\d,]+\.?\d*")
# Matches salary with optional spaces: $ 6 8,927.00 or $68,927.00
SALARY_TEXT_RE = re.compile(r"\$\s*[\d\s,]+\.\d{2}")

def extract_pdf_text(year, url):
    """
    Parse MLSPA PDFs that use plain text layout (2018-2020).
    Format: Club LastName FirstName Pos $ Base $ Guaranteed
    """
    print(f"  {year}: downloading PDF (text mode)...")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = []
    known_clubs = set(CLUB_MAP.keys()) | set(CLUB_MAP.values())

    with pdfplumber.open(io.BytesIO(r.content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                # Find salary amounts (with possible spaces inside)
                salaries_raw = SALARY_TEXT_RE.findall(line)
                if len(salaries_raw) < 2:
                    continue
                salary_vals = [parse_salary_str(s) for s in salaries_raw]
                salary_vals = [v for v in salary_vals if v and v > 0]
                if len(salary_vals) < 2:
                    continue
                # Find club name at start of line
                club = None
                for known in sorted(known_clubs, key=len, reverse=True):
                    if line.strip().startswith(known):
                        club = normalize_club(known)
                        break
                if not club:
                    # Try partial match anywhere in line prefix
                    prefix = line.split("$")[0]
                    for known in sorted(known_clubs, key=len, reverse=True):
                        if known in prefix:
                            club = normalize_club(known)
                            break
                if club:
                    rows.append({
                        "year": year,
                        "club": club,
                        "base_salary": salary_vals[0],
                        "guaranteed_comp": salary_vals[-1],
                    })
    print(f"    {len(rows)} records extracted")
    return pd.DataFrame(rows)


def extract_pdf_tables(year, url):
    """
    Parse MLSPA salary PDFs. Table columns are typically:
    Last Name | First Name | Club | Position | Base Salary | Guaranteed Comp
    We identify the club column by checking against our CLUB_MAP.
    """
    print(f"  {year}: downloading PDF...")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = []
    known_clubs = set(CLUB_MAP.keys()) | set(CLUB_MAP.values())

    with pdfplumber.open(io.BytesIO(r.content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if not row or len(row) < 4:
                        continue
                    cells = [str(c).strip() if c else "" for c in row]
                    # Find salary cells
                    salary_vals = []
                    salary_idxs = []
                    for i, c in enumerate(cells):
                        if SALARY_RE.match(c):
                            salary_vals.append(parse_salary_str(c))
                            salary_idxs.append(i)
                    if len(salary_vals) < 2:
                        continue
                    # Find club cell — must be a known club name
                    club = None
                    for c in cells:
                        if c in known_clubs:
                            club = normalize_club(c)
                            break
                    if not club:
                        continue
                    rows.append({
                        "year": year,
                        "club": club,
                        "base_salary": salary_vals[0],
                        "guaranteed_comp": salary_vals[-1],
                    })
    print(f"    {len(rows)} records extracted")
    return pd.DataFrame(rows)


# ── 3. Run collection ─────────────────────────────────────────────────────────
all_dfs = []
for year in sorted(SOURCES.keys()):
    fmt, url = SOURCES[year]
    try:
        if fmt == "csv":
            df = load_csv(year, url)
        elif fmt == "pdf_txt":
            df = extract_pdf_text(year, url)
        else:
            df = extract_pdf_tables(year, url)
        all_dfs.append(df)
    except Exception as e:
        print(f"    ERROR {year}: {e}")
    time.sleep(1)

EXCLUDE_CLUBS = {"MLS Pool", "Retired", "San Diego FC"}
players = pd.concat(all_dfs, ignore_index=True)
players = players[players["guaranteed_comp"] > 0]
players = players[~players["club"].isin(EXCLUDE_CLUBS)]
players.to_csv(OUT_DIR / "mlspa_players.csv", index=False)
print(f"\nSaved mlspa_players.csv — {players.shape}")

# ── 4. Aggregate to club-season ───────────────────────────────────────────────
agg = (players.groupby(["club", "year"])
              .agg(
                  player_count=("guaranteed_comp", "count"),
                  total_payroll=("guaranteed_comp", "sum"),
                  avg_salary=("guaranteed_comp", "mean"),
                  median_salary=("guaranteed_comp", "median"),
                  max_salary=("guaranteed_comp", "max"),
              )
              .round(0)
              .reset_index())

agg.to_csv(OUT_DIR / "mlspa_salaries.csv", index=False)
print(f"Saved mlspa_salaries.csv — {agg.shape}")

print("\n=== Club-Season Payroll Summary ===")
print(agg[["total_payroll","avg_salary","max_salary"]].describe().round(0))
print("\nTop 5 payrolls:")
print(agg.nlargest(5, "total_payroll")[["club","year","total_payroll","player_count"]].to_string(index=False))
