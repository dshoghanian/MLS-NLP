"""
MLS NLP Pipeline
================
End-to-end pipeline for MLS article collection, NLP enrichment,
network construction, and narrative analysis (2018–2024).

Stages:
  1. collector      — fetch articles from GDELT and RSS feeds
  2. enricher       — NLP entity/event/sentiment extraction
  3. network_builder— build co-occurrence graphs per time window
  4. analyzer       — centrality and narrative momentum metrics
"""

__version__ = "0.1.0"
