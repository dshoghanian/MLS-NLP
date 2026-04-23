"""
MLS Network Explorer
====================
Interactive Streamlit app to explore the MLS co-occurrence network.
Search any club, player, coach, or brand to see their connections.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ── Constants ──────────────────────────────────────────────────────────────────

ROOT            = Path(__file__).resolve().parent
NET_DIR         = ROOT / "data" / "press"  / "networks"
REDDIT_NET_DIR  = ROOT / "data" / "reddit" / "networks"
DATA_DIR        = ROOT / "data"

# ── Palette (Gephi-inspired dark) ─────────────────────────────────────────
PAGE_BG   = "#0d1117"   # GitHub dark — deep with blue undertone
CANVAS_BG = "#0d1117"   # same: network fills the page
CARD_BG   = "#161b22"   # elevated surface
SEPARATOR = "#30363d"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
ACCENT    = "#58a6ff"   # GitHub dark blue
FONT_STACK = "'Inter', 'Segoe UI', system-ui, sans-serif"

CLUB_COLORS = {
    "Atlanta United FC":      "#E4002B",
    "Austin FC":              "#00B140",
    "CF Montreal":            "#4169E1",
    "Charlotte FC":           "#1A9AE8",
    "Chicago Fire FC":        "#FF2A2A",
    "Colorado Rapids":        "#C8213F",
    "Columbus Crew":          "#FDD20E",
    "D.C. United":            "#DDDDDD",
    "FC Dallas":              "#E8173F",
    "Houston Dynamo FC":      "#FF7F00",
    "Inter Miami CF":         "#F7B5CD",
    "LA Galaxy":              "#4A7FCC",
    "LAFC":                   "#C8AA72",
    "Minnesota United FC":    "#8CD2F4",
    "Nashville SC":           "#ECE83A",
    "New England Revolution": "#3A7FCC",
    "New York City FC":       "#6CACE4",
    "New York Red Bulls":     "#ED3347",
    "Orlando City SC":        "#9B59B6",
    "Philadelphia Union":     "#3A6FCC",
    "Portland Timbers":       "#00A651",
    "Real Salt Lake":         "#E4002B",
    "San Jose Earthquakes":   "#2E86C1",
    "Seattle Sounders FC":    "#5BA135",
    "Sporting Kansas City":   "#91B0D5",
    "St. Louis City SC":      "#E4002B",
    "Toronto FC":             "#E4002B",
    "Vancouver Whitecaps FC": "#00B4D8",
}

NODE_TYPE_COLORS = {
    "club":      None,
    "player":    "#58a6ff",   # blue
    "coach":     "#e3b341",   # amber
    "executive": "#bc8cff",   # purple
    "brand":     "#3fb950",   # green
}

NODE_TYPE_SHAPES = {
    "club":      "dot",
    "player":    "diamond",
    "coach":     "triangle",
    "executive": "triangleDown",
    "brand":     "star",
}


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data
def get_available_windows(source: str = "Press") -> list[str]:
    base = NET_DIR if source != "Reddit" else REDDIT_NET_DIR
    return sorted(
        p.name for p in base.iterdir()
        if p.is_dir() and (p / f"{p.name}_club_cooccurrence.json").exists()
    )


def _load_graph(net_dir: Path, window: str, kind: str) -> nx.Graph:
    path = net_dir / window / f"{window}_{kind}.json"
    if not path.exists():
        return nx.Graph()
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


@st.cache_data
def load_cooccurrence(window: str) -> nx.Graph:
    return _load_graph(NET_DIR, window, "club_cooccurrence")


@st.cache_data
def load_reddit_cooccurrence(window: str) -> nx.Graph:
    return _load_graph(REDDIT_NET_DIR, window, "club_cooccurrence")


@st.cache_data
def load_entity(window: str) -> nx.Graph:
    return _load_graph(NET_DIR, window, "club_entity")


@st.cache_data
def load_reddit_entity(window: str) -> nx.Graph:
    return _load_graph(REDDIT_NET_DIR, window, "club_entity")


@st.cache_data
def load_brand_data() -> pd.DataFrame:
    path = DATA_DIR / "analysis" / "press" / "brand_earned_media.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_centrality() -> pd.DataFrame:
    path = DATA_DIR / "analysis" / "press" / "centrality_club_cooccurrence.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_reddit_centrality() -> pd.DataFrame:
    path = DATA_DIR / "analysis" / "reddit" / "centrality_club_reddit.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_press_vs_reddit() -> pd.DataFrame:
    path = DATA_DIR / "analysis" / "comparison" / "press_vs_reddit_centrality.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_master_summary() -> pd.DataFrame:
    path = DATA_DIR / "analysis" / "shared" / "master_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    window: str,
    show_players: bool,
    show_coaches: bool,
    show_executives: bool,
    show_brands: bool,
    min_edge_weight: int,
    max_person_nodes: int,
    source: str = "Press",
) -> nx.Graph:
    """Assemble a combined graph from co-occurrence + entity + brand data."""

    def _base_graph(loader_fn) -> nx.Graph:
        G = loader_fn(window)
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                           if d.get("weight", 1) < min_edge_weight]
        G.remove_edges_from(edges_to_remove)
        G.remove_nodes_from(list(nx.isolates(G)))
        for node in list(G.nodes()):
            G.nodes[node]["entity_type"] = "club"
        return G

    if source == "Press":
        G = _base_graph(load_cooccurrence)
    elif source == "Reddit":
        G = _base_graph(load_reddit_cooccurrence)
    else:  # Both — merge press + reddit, summing edge weights
        G_press  = _base_graph(load_cooccurrence)
        G_reddit = _base_graph(load_reddit_cooccurrence)
        G = nx.Graph()
        for n, d in G_press.nodes(data=True):
            G.add_node(n, **d)
        for n, d in G_reddit.nodes(data=True):
            if n not in G:
                G.add_node(n, **d)
        for u, v, d in G_press.edges(data=True):
            G.add_edge(u, v, weight=d.get("weight", 1), press_weight=d.get("weight", 1), reddit_weight=0)
        for u, v, d in G_reddit.edges(data=True):
            w = d.get("weight", 1)
            if G.has_edge(u, v):
                G[u][v]["weight"]        += w
                G[u][v]["reddit_weight"]  = w
            else:
                G.add_edge(u, v, weight=w, press_weight=0, reddit_weight=w)

    # Add person nodes from entity graph (press only — Reddit entity NER unreliable)
    if show_players or show_coaches or show_executives:
        G_entity = load_entity(window) if source != "Reddit" else load_reddit_entity(window)

        # Collect persons ranked by degree (most connected first)
        person_nodes = [
            (n, G_entity.nodes[n])
            for n in G_entity.nodes()
            if G_entity.nodes[n].get("entity_type") != "club"
        ]
        person_nodes.sort(key=lambda x: G_entity.degree(x[0], weight="weight"), reverse=True)

        added = 0
        for person, attrs in person_nodes:
            role = attrs.get("entity_type", "player")
            if role == "player"    and not show_players:    continue
            if role == "coach"     and not show_coaches:     continue
            if role == "executive" and not show_executives: continue
            if added >= max_person_nodes:
                break

            # Add edges from this person to clubs in G
            person_edges = [
                (u, v, d) for u, v, d in G_entity.edges(person, data=True)
                if (u in G.nodes() or v in G.nodes())
            ]
            if not person_edges:
                continue

            G.add_node(person, entity_type=role)
            for u, v, d in person_edges:
                w = d.get("weight", 1)
                if w >= min_edge_weight:
                    G.add_edge(u, v, weight=w)
            added += 1

    # Add brand nodes
    if show_brands:
        brand_df = load_brand_data()
        if not brand_df.empty:
            # Filter to window's year(s)
            year_part = window.split("_")[0]
            if year_part.isdigit():
                brand_df = brand_df[brand_df["year"] == int(year_part)]

            if not brand_df.empty:
                brand_edges = (
                    brand_df.dropna(subset=["club"])
                    .groupby(["brand", "club"])
                    .agg(weight=("article_id", "count"),
                         avg_sentiment=("sentiment", "mean"))
                    .reset_index()
                )
                for _, row in brand_edges.iterrows():
                    if row["club"] not in G.nodes():
                        continue
                    w = int(row["weight"])
                    if w < min_edge_weight:
                        continue
                    brand = row["brand"]
                    if brand not in G.nodes():
                        G.add_node(brand, entity_type="brand",
                                   avg_sentiment=round(row["avg_sentiment"], 3))
                    G.add_edge(brand, row["club"], weight=w)

    return G


# ── Graph annotator ───────────────────────────────────────────────────────────

def annotate_graph(G: nx.Graph, window: str, source: str = "Press") -> nx.Graph:
    """Attach centrality + performance data to graph node attributes."""
    import unicodedata

    def _normalize(s: str) -> str:
        return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()

    year        = window.split("_")[0]
    node_lookup = {_normalize(n): n for n in G.nodes()}

    def _resolve(csv_name: str):
        return node_lookup.get(_normalize(csv_name),
                               csv_name if csv_name in G.nodes() else None)

    # ── Centrality from the appropriate source ────────────────────────────────
    if source in ("Press", "Both"):
        cent = load_centrality()
        if not cent.empty:
            sub = cent[cent["time_window"] == window]
            if sub.empty and year.isdigit():
                sub = cent[cent["time_window"] == year]
            for _, row in sub.iterrows():
                n = _resolve(row.get("entity", ""))
                if n:
                    G.nodes[n]["press_pagerank"]  = row.get("pagerank")
                    G.nodes[n]["press_momentum"]   = row.get("momentum_label")

    if source in ("Reddit", "Both"):
        rcent = load_reddit_centrality()
        if not rcent.empty:
            sub = rcent[rcent["time_window"] == window]
            if sub.empty and year.isdigit():
                sub = rcent[rcent["time_window"] == year]
            for _, row in sub.iterrows():
                n = _resolve(row.get("entity", ""))
                if n:
                    G.nodes[n]["reddit_pagerank"] = row.get("pagerank")
                    G.nodes[n]["reddit_momentum"]  = row.get("momentum_label")

    # Set the "primary" pagerank + momentum based on active source
    for n in G.nodes():
        if source == "Press":
            G.nodes[n]["pagerank"]       = G.nodes[n].get("press_pagerank")
            G.nodes[n]["momentum_label"] = G.nodes[n].get("press_momentum", "")
        elif source == "Reddit":
            G.nodes[n]["pagerank"]       = G.nodes[n].get("reddit_pagerank")
            G.nodes[n]["momentum_label"] = G.nodes[n].get("reddit_momentum", "")
        else:  # Both — average the two
            pp = G.nodes[n].get("press_pagerank")  or 0
            rp = G.nodes[n].get("reddit_pagerank") or 0
            G.nodes[n]["pagerank"]       = (pp + rp) / 2 if (pp or rp) else None
            G.nodes[n]["momentum_label"] = G.nodes[n].get("press_momentum", "")

    # ── Cross-source comparison ranks (for Both mode) ─────────────────────────
    if source == "Both" and year.isdigit():
        pvr = load_press_vs_reddit()
        if not pvr.empty:
            sub = pvr[pvr["year"] == int(year)]
            for _, row in sub.iterrows():
                n = _resolve(row.get("entity", ""))
                if n:
                    G.nodes[n]["press_rank"]  = row.get("press_rank")
                    G.nodes[n]["reddit_rank"] = row.get("reddit_rank")
                    G.nodes[n]["rank_gap"]    = row.get("rank_gap")

    # ── Performance (season record, common to all modes) ──────────────────────
    summary = load_master_summary()
    if not summary.empty and year.isdigit():
        sub = summary[summary["year"] == int(year)]
        for _, row in sub.iterrows():
            n = _resolve(row.get("entity", ""))
            if n:
                G.nodes[n]["narrative_rank"]   = row.get("narrative_rank")
                G.nodes[n]["performance_rank"] = row.get("performance_rank")
                G.nodes[n]["points"]           = row.get("points")
                G.nodes[n]["cup_result"]       = row.get("cup_result")
                G.nodes[n]["made_playoffs"]    = row.get("made_playoffs")

    return G


# ── Pyvis renderer ────────────────────────────────────────────────────────────

def render_network(
    G: nx.Graph,
    search_term: str,
    hop_depth: int,
    height: int = 720,
    source: str = "Press",
) -> str:
    """
    Gephi-inspired dark network renderer.
    - No borders, strong color glow = professional graph look
    - Gossamer edges (ultra-thin, near-transparent) = complex web feel
    - Labels only on major hubs (top 40% by degree) = readable, not cluttered
    - Rich narrative tooltips = answers "what does this mean?"
    - physics=False, pre-computed layout = zero shaking
    """
    import math, statistics

    if G.number_of_nodes() == 0:
        return (
            f"<div style='background:{CANVAS_BG};height:{height}px;display:flex;"
            f"align-items:center;justify-content:center;'>"
            f"<p style='color:{TEXT_SEC};font-family:{FONT_STACK};font-size:15px;margin:0'>"
            f"No network data for this window.</p></div>"
        )

    # ── Determine visible nodes ──────────────────────────────────────────────
    term         = search_term.strip().lower()
    search_nodes : set[str] = set()
    direct_hits  : set[str] = set()
    if term:
        hits = [n for n in G.nodes() if term in n.lower()]
        if hits:
            direct_hits   = set(hits)
            search_nodes  = set(hits)
            for _ in range(hop_depth):
                nbrs: set[str] = set()
                for n in search_nodes:
                    nbrs.update(G.neighbors(n))
                search_nodes.update(nbrs)

    show_nodes = search_nodes if search_nodes else set(G.nodes())
    subgraph   = G.subgraph(show_nodes)

    # ── Pre-compute stable layout ────────────────────────────────────────────
    pos   = nx.spring_layout(subgraph, seed=42, k=2.5, iterations=150, weight="weight")
    SCALE = 850

    # ── Degree / weight stats ────────────────────────────────────────────────
    degree  = dict(G.degree(weight="weight"))
    max_deg = max(degree.values()) if degree else 1
    all_w   = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_w   = max(all_w) if all_w else 1

    # Label threshold — only label top 40% by weighted degree
    deg_values   = sorted(degree.values())
    label_thresh = deg_values[int(len(deg_values) * 0.60)] if deg_values else 0

    # ── Build pyvis + tooltip data dict ──────────────────────────────────────
    # vis.js renders `title` as plain text (not HTML). We store tooltip HTML in
    # a JS dict and use a custom hoverNode listener that sets innerHTML.
    net = Network(height=f"{height}px", width="100%",
                  bgcolor=CANVAS_BG, font_color=TEXT_PRI)

    tooltip_data: dict[str, str] = {}      # node_id  → HTML string
    edge_tooltip_data: dict[str, str] = {} # "u||v"   → HTML string

    for node in show_nodes:
        if node not in G.nodes():
            continue

        attrs     = G.nodes[node]
        node_type = attrs.get("entity_type", "club")
        shape     = NODE_TYPE_SHAPES.get(node_type, "dot")
        is_club   = node_type == "club"
        is_hit    = node in direct_hits
        raw_deg   = degree.get(node, 1)

        # ── Color ─────────────────────────────────────────────────────────
        fill = (CLUB_COLORS.get(node, "#58a6ff")
                if is_club else NODE_TYPE_COLORS.get(node_type, "#58a6ff"))

        # ── Size: dramatic log scale (hubs are visually dominant) ─────────
        t    = math.log1p(raw_deg) / math.log1p(max_deg)  # 0–1
        size = 8 + 38 * t                                  # 8 → 46
        if is_hit:
            size = max(size + 12, 30)

        # ── Label: only for major hubs + search hits ──────────────────────
        show_label = raw_deg >= label_thresh or is_hit
        label = node if show_label else ""

        # ── Glow: node's own color as shadow (the Gephi signature look) ───
        glow_size = int(8 + 20 * t)
        if is_hit:
            glow_size = 30

        # ── Build tooltip HTML (injected via custom JS, not vis.js title) ──
        nr     = attrs.get("narrative_rank")
        pr     = attrs.get("pagerank")
        mom    = attrs.get("momentum_label", "")
        perf_r = attrs.get("performance_rank")
        pts    = attrs.get("points")
        result = attrs.get("cup_result", "")
        n_conn = G.degree(node)

        icon = ""
        mom_icon = "+" if mom == "rising" else ("-" if mom == "falling" else "~")

        cup_map = {
            "MLS Cup Winner":     "MLS Cup Champion",
            "MLS Cup Runner-up":  "MLS Cup Final",
            "conference_final":   "Conference Final",
            "conference_semifinal": "Conference Semifinal",
            "first_round":        "First Round Exit",
            "Missed Playoffs":    "Missed Playoffs",
        }
        result_label = cup_map.get(result, result.replace("_", " ").title() if result else "—")

        top_nbrs = sorted(
            [(nb, G[node][nb].get("weight", 0)) for nb in G.neighbors(node)],
            key=lambda x: x[1], reverse=True
        )[:3]
        nbr_rows = "".join(
            f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>{nb}</td>"
            f"<td style='color:{ACCENT};font-weight:600'>{w}</td></tr>"
            for nb, w in top_nbrs
        )

        src_label = {"Press": "Press", "Reddit": "Reddit", "Both": "Press & Reddit"}[source]

        tip = (
            f"<div style=\"font-family:'Inter',system-ui,sans-serif;font-size:12px;"
            f"color:{TEXT_PRI};line-height:1.6;min-width:220px\">"
            f"<div style='font-size:14px;font-weight:700;margin-bottom:4px;"
            f"border-bottom:1px solid {SEPARATOR};padding-bottom:6px'>"
            f"{node}</div>"
            f"<div style='font-size:10px;color:{TEXT_SEC};margin-bottom:6px'>"
            f"Source: {src_label}</div>"
        )

        # Narrative block (clubs only)
        if is_club:
            if source == "Both":
                p_rank = attrs.get("press_rank")
                r_rank = attrs.get("reddit_rank")
                gap    = attrs.get("rank_gap")
                pp     = attrs.get("press_pagerank")
                rp     = attrs.get("reddit_pagerank")
                if gap is not None:
                    if gap > 2:
                        gap_txt = f"<span style='color:#e3b341'>Press rates higher (+{int(gap)})</span>"
                    elif gap < -2:
                        gap_txt = f"<span style='color:#3fb950'>Fans rate higher ({abs(int(gap))} spots)</span>"
                    else:
                        gap_txt = f"<span style='color:{TEXT_SEC}'>Aligned between press & fans</span>"
                else:
                    gap_txt = ""
                tip += (
                    f"<div style='margin-bottom:6px'>"
                    f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:.07em;"
                    f"color:{TEXT_SEC};margin-bottom:3px'>Narrative comparison</div>"
                    f"<table style='border-collapse:collapse;width:100%'>"
                    f"<tr><td style='padding:1px 8px 1px 0;color:{TEXT_SEC}'>Press rank</td>"
                    f"<td style='font-weight:700;color:{ACCENT}'>{'#'+str(int(p_rank)) if p_rank else '—'}</td>"
                    f"<td style='padding:1px 8px 1px 8px;color:{TEXT_SEC}'>PageRank</td>"
                    f"<td style='font-weight:600'>{f'{float(pp):.4f}' if pp else '—'}</td></tr>"
                    f"<tr><td style='padding:1px 8px 1px 0;color:{TEXT_SEC}'>Reddit rank</td>"
                    f"<td style='font-weight:700;color:#3fb950'>{'#'+str(int(r_rank)) if r_rank else '—'}</td>"
                    f"<td style='padding:1px 8px 1px 8px;color:{TEXT_SEC}'>PageRank</td>"
                    f"<td style='font-weight:600'>{f'{float(rp):.4f}' if rp else '—'}</td></tr>"
                    f"</table>"
                    + (f"<div style='margin-top:4px;font-size:11px'>{gap_txt}</div>" if gap_txt else "")
                    + f"</div>"
                )
            elif pr and nr:
                tip += (
                    f"<div style='margin-bottom:6px'>"
                    f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:.07em;"
                    f"color:{TEXT_SEC};margin-bottom:3px'>Media narrative</div>"
                    f"<table style='border-collapse:collapse;width:100%'>"
                    f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>Narrative rank</td>"
                    f"<td style='color:{ACCENT};font-weight:700'>#{int(nr)} of 28 &nbsp;{mom_icon}</td></tr>"
                    f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>PageRank score</td>"
                    f"<td style='font-weight:600'>{float(pr):.4f}</td></tr>"
                    f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>Momentum</td>"
                    f"<td style='font-weight:600'>{mom.capitalize() if mom else '—'}</td></tr>"
                    f"</table></div>"
                )

        # Season performance block (clubs only)
        if is_club and perf_r:
            tip += (
                f"<div style='margin-bottom:6px;border-top:1px solid {SEPARATOR};padding-top:6px'>"
                f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:.07em;"
                f"color:{TEXT_SEC};margin-bottom:3px'>Season performance</div>"
                f"<table style='border-collapse:collapse;width:100%'>"
                f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>Points</td>"
                f"<td style='font-weight:700;font-size:13px'>{int(pts) if pts else '—'} pts</td></tr>"
                f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>Standing</td>"
                f"<td style='font-weight:600'>#{int(perf_r)} overall</td></tr>"
                f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>Best result</td>"
                f"<td style='font-weight:600'>{result_label}</td></tr>"
                f"</table></div>"
            )

        # Co-occurrence block (all nodes)
        doc_word = "posts" if source == "Reddit" else ("articles+posts" if source == "Both" else "articles")
        tip += (
            f"<div style='border-top:1px solid {SEPARATOR};padding-top:6px'>"
            f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:.07em;"
            f"color:{TEXT_SEC};margin-bottom:3px'>Co-occurrence</div>"
            f"<table style='border-collapse:collapse;width:100%'>"
            f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>Connected to</td>"
            f"<td style='font-weight:700'>{n_conn} clubs</td></tr>"
            f"<tr><td style='padding:1px 10px 1px 0;color:{TEXT_SEC}'>Total co-mentions</td>"
            f"<td style='font-weight:700'>{raw_deg} {doc_word}</td></tr>"
            f"</table>"
        )
        if nbr_rows:
            tip += (
                f"<div style='margin-top:5px;font-size:10px;text-transform:uppercase;"
                f"letter-spacing:.07em;color:{TEXT_SEC}'>Strongest links</div>"
                f"<table style='border-collapse:collapse;width:100%;margin-top:2px'>"
                f"{nbr_rows}</table>"
            )
        tip += "</div>"

        # Brand sentiment block
        sentiment = attrs.get("avg_sentiment")
        if sentiment is not None and not is_club:
            s_label = "Positive" if sentiment > 0.1 else ("Negative" if sentiment < -0.1 else "Neutral")
            tip += (
                f"<div style='margin-top:6px;border-top:1px solid {SEPARATOR};"
                f"padding-top:6px'>Brand sentiment: <b>{s_label}</b> ({sentiment:+.2f})</div>"
            )

        tip += "</div>"
        tooltip_data[node] = tip

        x, y = pos.get(node, (0, 0))
        net.add_node(
            node,
            label=label,
            title=" ",          # non-empty so vis.js enables hover; content replaced by JS
            color={
                "background": fill,
                "border":     fill,
                "highlight":  {"background": fill, "border": ACCENT},
                "hover":      {"background": fill, "border": ACCENT},
            },
            size=size,
            shape=shape,
            borderWidth=0,
            shadow={
                "enabled": True,
                "color":   fill,
                "size":    glow_size,
                "x": 0, "y": 0,
            },
            font={
                "size":   11 if is_club else 10,
                "color":  TEXT_PRI,
                "face":   "Inter, Segoe UI, system-ui, sans-serif",
                "strokeWidth": 2,
                "strokeColor": CANVAS_BG,
            },
            x=float(x * SCALE),
            y=float(y * SCALE),
            physics=False,
        )

    # ── Edges: gossamer web ───────────────────────────────────────────────────
    seen: set[tuple] = set()
    for u, v, data in G.edges(data=True):
        if u not in show_nodes or v not in show_nodes:
            continue
        key = tuple(sorted([u, v]))
        if key in seen:
            continue
        seen.add(key)

        w     = data.get("weight", 1)
        t     = w / max_w
        width = 0.3 + 2.2 * t

        alpha_f = 0.04 + 0.22 * t
        edge_c  = f"rgba(230,237,243,{alpha_f:.2f})"

        if direct_hits and (u in direct_hits or v in direct_hits):
            edge_c = f"rgba(88,166,255,{0.4 + 0.4 * t:.2f})"
            width  = width * 2

        edge_key = f"{u}||{v}"
        edge_tooltip_data[edge_key] = (
            f"<div style=\"font-family:'Inter',system-ui,sans-serif;font-size:12px;"
            f"color:{TEXT_PRI};line-height:1.6\">"
            f"<b>{u}</b> &harr; <b>{v}</b><br>"
            f"<span style='color:{TEXT_SEC}'>Co-mentioned in </span>"
            f"<b>{w}</b> article{'s' if w != 1 else ''}"
            f"</div>"
        )
        net.add_edge(u, v, value=w, title=" ", color=edge_c, width=width)

    # ── vis.js options ────────────────────────────────────────────────────────
    net.set_options("""
    {
      "physics": { "enabled": false },
      "interaction": {
        "hover": true,
        "tooltipDelay": 99999,
        "navigationButtons": false,
        "keyboard": false,
        "zoomSpeed": 0.6
      },
      "edges": {
        "smooth": { "enabled": true, "type": "curvedCW", "roundness": 0.18 },
        "selectionWidth": 3,
        "hoverWidth": 2
      },
      "nodes": { "scaling": { "min": 8, "max": 50 } }
    }
    """)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html = Path(f.name).read_text()

    # Expose the vis.js Network instance so our injected script can reach it.
    # pyvis generates `network = new vis.Network(` (no `var`).
    html = html.replace(
        "network = new vis.Network(",
        "window.mlsNetwork = new vis.Network("
    )

    # ── Inject CSS ────────────────────────────────────────────────────────────
    css = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {CANVAS_BG};
    font-family: 'Inter', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  #mynetwork {{
    border: 1px solid {SEPARATOR} !important;
    border-radius: 12px !important;
    background: {CANVAS_BG} !important;
  }}
  /* Hide the default vis.js tooltip — our custom one replaces it */
  div.vis-tooltip {{ display: none !important; }}

  #mls-tip {{
    position: fixed;
    z-index: 99999;
    display: none;
    background: rgba(13,17,23,0.97);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid {SEPARATOR};
    border-radius: 10px;
    color: {TEXT_PRI};
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 12px;
    padding: 12px 14px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.8);
    max-width: 275px;
    pointer-events: none;
    line-height: 1.5;
  }}
</style>
"""
    html = html.replace("</head>", css + "</head>")

    # ── Inject custom tooltip JavaScript ──────────────────────────────────────
    tip_json      = json.dumps(tooltip_data,      ensure_ascii=False)
    edge_tip_json = json.dumps(edge_tooltip_data, ensure_ascii=False)

    custom_js = f"""
<div id="mls-tip"></div>
<script>
(function() {{
  var nodeTips = {tip_json};
  var edgeTips = {edge_tip_json};
  var tip = document.getElementById('mls-tip');
  var mx = 0, my = 0;

  document.addEventListener('mousemove', function(e) {{
    mx = e.clientX; my = e.clientY;
    if (tip.style.display === 'block') positionTip();
  }});

  function positionTip() {{
    var x = mx + 18, y = my - 10;
    if (x + 285 > window.innerWidth)   x = mx - 293;
    if (y + tip.offsetHeight > window.innerHeight) y = my - tip.offsetHeight - 8;
    tip.style.left = x + 'px';
    tip.style.top  = y + 'px';
  }}

  function showTip(html) {{
    tip.innerHTML = html;
    tip.style.display = 'block';
    positionTip();
  }}

  function hideTip() {{ tip.style.display = 'none'; }}

  function waitForNetwork() {{
    if (!window.mlsNetwork) {{ setTimeout(waitForNetwork, 80); return; }}
    var net = window.mlsNetwork;

    net.on('hoverNode', function(p) {{
      var h = nodeTips[p.node];
      if (h) showTip(h);
    }});
    net.on('blurNode', hideTip);

    net.on('hoverEdge', function(p) {{
      var ed = net.body.data.edges.get(p.edge);
      if (!ed) return;
      var h = edgeTips[ed.from + '||' + ed.to] || edgeTips[ed.to + '||' + ed.from];
      if (h) showTip(h);
    }});
    net.on('blurEdge', hideTip);
  }}

  waitForNetwork();
}})();
</script>
"""
    html = html.replace("</body>", custom_js + "\n</body>")
    return html


# ── Node info panel ───────────────────────────────────────────────────────────

def show_node_info(node_name: str, G: nx.Graph, window: str):
    if node_name not in G.nodes():
        st.info(f"'{node_name}' not found in the {window} network.")
        return

    attrs     = G.nodes[node_name]
    node_type = attrs.get("entity_type", "unknown")
    neighbors = sorted(G.neighbors(node_name),
                       key=lambda n: G[node_name][n].get("weight", 0),
                       reverse=True)

    color = (CLUB_COLORS.get(node_name, "#aaaaaa")
             if node_type == "club"
             else NODE_TYPE_COLORS.get(node_type, "#aaaaaa"))

    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background:{color}20;"
        f"border-left:4px solid {color};margin-bottom:12px'>"
        f"<h3 style='margin:0'>{node_name}</h3>"
        f"<span style='opacity:0.7'>Type: {node_type} &nbsp;|&nbsp; "
        f"Window: {window} &nbsp;|&nbsp; "
        f"Connections: {G.degree(node_name)}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Centrality scores
    cent = load_centrality()
    if not cent.empty and "entity" in cent.columns:
        year_part = window.split("_")[0]
        row = cent[(cent["entity"] == node_name) & (cent["time_window"] == window)]
        if row.empty and year_part.isdigit():
            row = cent[(cent["entity"] == node_name) & (cent["time_window"] == year_part)]
        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PageRank",    f"{r.get('pagerank', 0):.4f}")
            c2.metric("Degree",      f"{r.get('degree', 0):.4f}")
            c3.metric("Eigenvector", f"{r.get('eigenvector', 0):.4f}")
            c4.metric("Betweenness", f"{r.get('betweenness', 0):.4f}")

    # Performance stats (clubs only)
    if node_type == "club":
        summary = load_master_summary()
        if not summary.empty:
            year_part = window.split("_")[0]
            if year_part.isdigit():
                row = summary[(summary["entity"] == node_name) &
                              (summary["year"] == int(year_part))]
                if not row.empty:
                    r = row.iloc[0]
                    st.markdown("**Season performance**")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Points",            r.get("points", "—"))
                    p2.metric("Narrative Rank",    f"#{r.get('narrative_rank', '—')}")
                    p3.metric("Performance Rank",  f"#{r.get('performance_rank', '—')}")

    # Top connections
    st.markdown("**Top connections by co-mention weight**")
    rows = []
    for nb in neighbors[:12]:
        w = G[node_name][nb].get("weight", 0)
        nb_type = G.nodes[nb].get("entity_type", "?")
        rows.append({"Node": nb, "Type": nb_type, "Co-mentions": w})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Brand info
    if node_type == "brand":
        brand_df = load_brand_data()
        if not brand_df.empty:
            sub = brand_df[brand_df["brand"] == node_name]
            if not sub.empty:
                st.markdown("**Brand earned media stats**")
                b1, b2, b3 = st.columns(3)
                b1.metric("Total article mentions", len(sub["article_id"].unique()))
                b2.metric("Avg sentiment",          f"{sub['sentiment'].mean():.2f}")
                b3.metric("Top associated club",
                          sub.groupby("club").size().idxmax() if sub["club"].notna().any() else "—")


# ── Main app ──────────────────────────────────────────────────────────────────

def main():  # noqa: C901
    st.set_page_config(
        page_title="MLS Network Explorer",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        /* ── Base ─────────────────────────────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: {FONT_STACK} !important;
            -webkit-font-smoothing: antialiased;
        }}
        .stApp {{
            background: {PAGE_BG} !important;
        }}
        .block-container {{
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem;
            max-width: 100% !important;
        }}
        /* Hide Streamlit's default top decoration bar */
        header[data-testid="stHeader"] {{
            display: none !important;
        }}

        /* Force all text to be light on dark backgrounds */
        .stApp, .stApp * {{
            color: {TEXT_PRI};
        }}

        /* ── Sidebar ──────────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background: {CANVAS_BG} !important;
            border-right: 1px solid {SEPARATOR} !important;
        }}
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] .stMarkdown {{
            color: {TEXT_PRI} !important;
        }}
        /* Muted sidebar text gets secondary colour via inline HTML */

        /* ── Expander ─────────────────────────────────────────────────────── */
        div[data-testid="stExpander"] {{
            border: 1px solid {SEPARATOR} !important;
            border-radius: 12px !important;
            background: {CARD_BG} !important;
        }}
        div[data-testid="stExpander"] *,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] p {{
            color: {TEXT_PRI} !important;
        }}

        /* ── Metric cards ─────────────────────────────────────────────────── */
        div[data-testid="metric-container"] {{
            background: {CARD_BG} !important;
            border: 1px solid {SEPARATOR} !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
        }}
        .stMetric label {{
            font-size: 0.70rem !important;
            color: {TEXT_SEC} !important;
            text-transform: uppercase !important;
            letter-spacing: .06em !important;
            font-weight: 500 !important;
        }}
        .stMetric [data-testid="stMetricValue"] {{
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            color: {TEXT_PRI} !important;
        }}

        /* ── Text input ───────────────────────────────────────────────────── */
        div[data-testid="stTextInput"] input {{
            border-radius: 10px !important;
            border: 1px solid {SEPARATOR} !important;
            background: {CARD_BG} !important;
            color: {TEXT_PRI} !important;
            font-family: {FONT_STACK} !important;
        }}
        div[data-testid="stTextInput"] input::placeholder {{
            color: {TEXT_SEC} !important;
        }}
        div[data-testid="stTextInput"] input:focus {{
            border-color: {ACCENT} !important;
            box-shadow: 0 0 0 3px rgba(10,132,255,0.25) !important;
            outline: none !important;
        }}

        /* ── Radio buttons ────────────────────────────────────────────────── */
        div[data-testid="stRadio"] label {{
            color: {TEXT_PRI} !important;
        }}

        /* ── Checkboxes ───────────────────────────────────────────────────── */
        div[data-testid="stCheckbox"] label {{
            color: {TEXT_PRI} !important;
        }}

        /* ── Selectbox / slider ───────────────────────────────────────────── */
        div[data-testid="stSelectbox"] * {{
            color: {TEXT_PRI} !important;
            background: {CARD_BG} !important;
        }}
        div[data-testid="stSlider"] * {{
            color: {TEXT_PRI} !important;
        }}

        /* ── Divider ──────────────────────────────────────────────────────── */
        hr {{
            border-color: {SEPARATOR} !important;
            margin: 12px 0 !important;
        }}

        /* ── Info / warning boxes ─────────────────────────────────────────── */
        div[data-testid="stInfo"],
        div.stAlert {{
            background: {CARD_BG} !important;
            border: 1px solid {SEPARATOR} !important;
            border-radius: 10px !important;
            color: {TEXT_PRI} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<p style='font-size:0.65rem;color:{TEXT_SEC};letter-spacing:.1em;"
            f"text-transform:uppercase;margin:0 0 2px 0'>MLS · 2018–2024</p>"
            f"<h2 style='font-size:1.05rem;font-weight:700;margin:0 0 4px 0'>"
            f"Narrative Network</h2>"
            f"<p style='font-size:0.75rem;color:{TEXT_SEC};margin:0 0 14px 0'>"
            f"Who gets covered with whom — and how much</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        with st.expander("How to read this", expanded=False):
            st.markdown(
                "**Each circle = a club** in its team colour. "
                "Bigger = more narrative presence.\n\n"
                "**Lines** connect clubs co-mentioned in the same article or post. "
                "Brighter and thicker = more shared coverage.\n\n"
                "**Hover** any node for a full breakdown. "
                "**Search** to zoom into one club's connections."
            )

        st.divider()

        # ── Source toggle ─────────────────────────────────────────────────────
        st.markdown(f"<p style='font-size:0.65rem;color:{TEXT_SEC};letter-spacing:.08em;"
                    f"text-transform:uppercase;margin-bottom:6px'>Source</p>",
                    unsafe_allow_html=True)
        source = st.radio(
            "Source",
            options=["Press", "Reddit", "Both"],
            index=0,
            label_visibility="collapsed",
            help="Press = news articles · Reddit = fan posts · Both = merged network",
        )
        if source == "Press":
            st.markdown(
                f"<p style='font-size:0.72rem;color:{TEXT_SEC};margin-top:3px'>"
                f"News articles, 2018–2024</p>", unsafe_allow_html=True)
        elif source == "Reddit":
            st.markdown(
                f"<p style='font-size:0.72rem;color:{TEXT_SEC};margin-top:3px'>"
                f"Fan posts &amp; comments, 2018–2024</p>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<p style='font-size:0.72rem;color:{TEXT_SEC};margin-top:3px'>"
                f"Edge weights combined. Hover nodes to compare ranks.</p>",
                unsafe_allow_html=True)

        st.divider()

        # ── Search ───────────────────────────────────────────────────────────
        st.markdown(f"<p style='font-size:0.65rem;color:{TEXT_SEC};letter-spacing:.08em;"
                    f"text-transform:uppercase;margin-bottom:6px'>Search</p>",
                    unsafe_allow_html=True)
        search_term = st.text_input("Search",
            placeholder="Inter Miami, Messi, Subaru…",
            label_visibility="collapsed")
        hop_depth = st.radio("Depth", options=[1, 2],
            format_func=lambda x: "Direct connections" if x == 1 else "2 degrees out",
            label_visibility="collapsed")

        st.divider()

        # ── Time window ──────────────────────────────────────────────────────
        st.markdown(f"<p style='font-size:0.65rem;color:{TEXT_SEC};letter-spacing:.08em;"
                    f"text-transform:uppercase;margin-bottom:6px'>Time Window</p>",
                    unsafe_allow_html=True)
        windows   = get_available_windows(source)
        yearly    = [w for w in windows if "_" not in w]
        quarterly = [w for w in windows if "_Q" in w]
        monthly   = [w for w in windows if "_" in w and "_Q" not in w]

        gran_opts = ["Yearly", "Quarterly"] + (["Monthly"] if monthly else [])
        granularity    = st.selectbox("Gran", gran_opts, label_visibility="collapsed")
        window_options = (yearly if granularity == "Yearly"
                          else quarterly if granularity == "Quarterly" else monthly)
        window = st.select_slider("Win", options=window_options,
                                  value=window_options[-1] if window_options else None,
                                  label_visibility="collapsed")
        st.divider()

        # ── Layers ───────────────────────────────────────────────────────────
        st.markdown(f"<p style='font-size:0.65rem;color:{TEXT_SEC};letter-spacing:.08em;"
                    f"text-transform:uppercase;margin-bottom:6px'>Layers</p>",
                    unsafe_allow_html=True)
        show_clubs      = st.checkbox("Clubs",      value=True)
        show_players    = st.checkbox("Players",    value=False)
        show_coaches    = st.checkbox("Coaches",    value=False)
        show_executives = st.checkbox("Executives", value=False)
        show_brands     = st.checkbox("Brands",     value=False and source == "Press")
        max_persons = (st.slider("Max people", 5, 60, 20, 5, label_visibility="collapsed")
                       if (show_players or show_coaches or show_executives) else 20)
        st.divider()

        # ── Min edge ─────────────────────────────────────────────────────────
        doc_label = "posts" if source == "Reddit" else "articles"
        st.markdown(f"<p style='font-size:0.65rem;color:{TEXT_SEC};letter-spacing:.08em;"
                    f"text-transform:uppercase;margin-bottom:6px'>Min shared {doc_label}</p>",
                    unsafe_allow_html=True)
        min_weight = st.slider("Min", 1, 20, 2, label_visibility="collapsed",
                               help="Raise to show only the strongest connections")
        st.divider()

        # ── Legend ───────────────────────────────────────────────────────────
        st.markdown(f"<p style='font-size:0.65rem;color:{TEXT_SEC};letter-spacing:.08em;"
                    f"text-transform:uppercase;margin-bottom:8px'>Legend</p>",
                    unsafe_allow_html=True)
        for sym, lbl, col in [
            ("●", "Club — team colour", TEXT_SEC),
            ("◆", "Player",    NODE_TYPE_COLORS["player"]),
            ("▲", "Coach",     NODE_TYPE_COLORS["coach"]),
            ("▽", "Executive", NODE_TYPE_COLORS["executive"]),
            ("*", "Brand",     NODE_TYPE_COLORS["brand"]),
        ]:
            st.markdown(
                f"<div style='margin-bottom:5px'>"
                f"<span style='color:{col}'>{sym}</span> "
                f"<span style='font-size:0.8rem'>{lbl}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Guard ─────────────────────────────────────────────────────────────────
    if not window:
        st.warning("No network data found. Run the pipeline first.")
        return

    # ── Build + annotate ─────────────────────────────────────────────────────
    with st.spinner(""):
        G = build_graph(window=window, show_players=show_players,
                        show_coaches=show_coaches, show_executives=show_executives,
                        show_brands=show_brands and source == "Press",
                        min_edge_weight=min_weight,
                        max_person_nodes=max_persons, source=source)
        if not show_clubs:
            G.remove_nodes_from([n for n, d in G.nodes(data=True)
                                  if d.get("entity_type") == "club"])
        G = annotate_graph(G, window, source=source)

    term          = search_term.strip().lower()
    matched_nodes = [n for n in G.nodes() if term in n.lower()] if term else []

    # ── Stat strip ────────────────────────────────────────────────────────────
    doc_word = "posts" if source == "Reddit" else "articles"
    focus_txt = f" · showing connections for <b>{search_term}</b>" if term else ""
    st.markdown(
        f"<div style='display:flex;gap:28px;align-items:baseline;"
        f"padding:10px 4px 6px;border-bottom:1px solid {SEPARATOR};margin-bottom:8px'>"
        f"<span style='font-size:0.72rem;color:{TEXT_SEC}'>"
        f"<b style='font-size:1.05rem;color:{TEXT_PRI}'>{G.number_of_nodes()}</b> clubs</span>"
        f"<span style='font-size:0.72rem;color:{TEXT_SEC}'>"
        f"<b style='font-size:1.05rem;color:{TEXT_PRI}'>{G.number_of_edges()}</b> co-mention links</span>"
        f"<span style='font-size:0.72rem;color:{TEXT_SEC}'>Season <b style='color:{TEXT_PRI}'>{window}</b></span>"
        f"<span style='font-size:0.72rem;color:{TEXT_SEC}'>Source: <b style='color:{TEXT_PRI}'>{source}</b></span>"
        + (f"<span style='font-size:0.72rem;color:{TEXT_SEC}'>{focus_txt}</span>" if term else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Full-width network ────────────────────────────────────────────────────
    components.html(render_network(G, search_term, hop_depth, source=source),
                    height=680, scrolling=False)

    st.markdown(
        f"<p style='font-size:0.7rem;color:{TEXT_SEC};margin-top:4px'>"
        f"Scroll to zoom · Drag to pan · Hover any node · "
        f"Labels visible on the most-connected clubs only</p>",
        unsafe_allow_html=True,
    )

    # ── Node detail / leaderboard ─────────────────────────────────────────────
    st.divider()
    if term and matched_nodes:
        exact        = [n for n in matched_nodes if n.lower() == term]
        node_to_show = exact[0] if exact else matched_nodes[0]
        if len(matched_nodes) > 1:
            node_to_show = st.selectbox(f"{len(matched_nodes)} matches", matched_nodes, index=0)
        show_node_info(node_to_show, G, window)
    elif term:
        st.info(f"**{search_term}** not found in this window — try a different year or enable more layers.")
    else:
        # Leaderboard from the active source
        cent_df = load_reddit_centrality() if source == "Reddit" else load_centrality()
        if not cent_df.empty:
            sub = cent_df[cent_df["time_window"] == window].nlargest(5, "pagerank")
            if sub.empty and "_" in window:
                sub = cent_df[cent_df["time_window"] == window.split("_")[0]].nlargest(5, "pagerank")
            if not sub.empty:
                hub_label = "Top 5 fan narrative hubs" if source == "Reddit" else "Top 5 narrative hubs"
                st.markdown(
                    f"<p style='font-size:0.65rem;color:{TEXT_SEC};letter-spacing:.08em;"
                    f"text-transform:uppercase;margin-bottom:12px'>{hub_label}</p>",
                    unsafe_allow_html=True,
                )
                cols = st.columns(5)
                for i, (_, r) in enumerate(sub.iterrows()):
                    club  = r["entity"]
                    color = CLUB_COLORS.get(club, TEXT_SEC)
                    with cols[i]:
                        st.markdown(
                            f"<div style='border-top:3px solid {color};padding:10px 0 4px'>"
                            f"<div style='font-size:0.62rem;color:{TEXT_SEC}'>#{i+1}</div>"
                            f"<div style='font-size:0.82rem;font-weight:600;line-height:1.3;"
                            f"margin:2px 0'>{club}</div>"
                            f"<div style='font-size:0.7rem;color:{TEXT_SEC}'>"
                            f"PageRank {r['pagerank']:.4f}</div></div>",
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
