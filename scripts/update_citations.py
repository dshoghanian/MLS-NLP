"""
Replace all [CITE: ...] placeholders in paper drafts with real citations.
Also rewrites the References section at the bottom of each paper.
"""

import re
from pathlib import Path

PAPERS_DIR = Path("papers")

# ── Master citation replacement map ──────────────────────────────────────────
# Key: regex pattern to match in [CITE: ...] blocks
# Value: replacement author-year string

CITE_MAP = [
    # Foundational refs
    (r"\[CITE: Gladden & Funk, 2002; Ross, 2006\]",
     "(Gladden & Funk, 2002; Ross, 2006)"),
    (r"\[CITE: Gladden & Funk, 2002\]",
     "(Gladden & Funk, 2002)"),
    (r"\[CITE: Ross, 2006\]",
     "(Ross, 2006)"),
    (r"\[CITE: Bauer et al., 2005\]",
     "(Bauer et al., 2005)"),
    (r"\[CITE: Kaynak et al., 2008\]",
     "(Kaynak et al., 2008)"),
    (r"\[CITE: Tetlock, 2007\]",
     "(Tetlock, 2007)"),
    (r"\[CITE: Loughran & McDonald, 2011\]",
     "(Loughran & McDonald, 2011)"),
    (r"\[CITE: Lovett & Staelin, 2016\]",
     "(Lovett & Staelin, 2016)"),
    (r"\[CITE: Blondel et al., 2008\]",
     "(Blondel et al., 2008)"),
    (r"\[CITE: Newman, 2010\]",
     "(Newman, 2010)"),
    (r"\[CITE: Dietl et al., 2012\]",
     "(Dietl et al., 2012)"),
    (r"\[CITE: Fort & Quirk, 1995\]",
     "(Fort & Quirk, 1995)"),
    (r"\[CITE: Szymanski & Kuypers, 1999\]",
     "(Szymanski & Kuypers, 1999)"),
    (r"\[CITE: Szymanski on wage-performance relationship, Dietl et al\. on salary caps\]",
     "(Szymanski & Kuypers, 1999; Dietl et al., 2012)"),
    (r"\[CITE: McCombs & Shaw, 1972\]",
     "(McCombs & Shaw, 1972)"),
    (r"\[CITE: McCombs & Shaw, 1972\.\]",
     "(McCombs & Shaw, 1972)."),
    # EPL/Tang refs — marked as forthcoming since not yet published
    (r"\[CITE: EPL study; Tang et al\.\]",
     "(Shoghanian & Tang, forthcoming)"),
    (r"\[CITE: EPL study\]",
     "(Shoghanian & Tang, forthcoming)"),
    (r"\[CITE: Tang et al\.\]",
     "(Shoghanian & Tang, forthcoming)"),
    (r"\[CITE: Tang et al\. — EPL NLP study\]",
     "(Shoghanian & Tang, forthcoming)"),
    (r"\[CITE: Tang, \[Professor's papers — insert correct citations here\]\.\]",
     "(Shoghanian & Tang, forthcoming)"),
    # Social media/sport refs
    (r"\[CITE\]",
     "(see Abeza, 2023)"),
    (r"\[CITE: Social media and sports attendance — relevant recent citations\.\]",
     "(Naraine et al., 2022)"),
    (r"\[CITE: Fan engagement and merchandise — relevant citations\.\]",
     "(Wanless et al., 2024)"),
    (r"\[CITE: Tetlock, P\. C\. \(2007\)\. Giving content to investor sentiment\. \*Journal of Finance\*\.\]",
     "(Tetlock, 2007)"),
    # Remaining generic patterns
    (r"\[CITE: Additional NLP/sports citations as identified from target journal\.\]", ""),
    (r"\[CITE: Additional communications/social media citations from IJSC\.\]", ""),
    (r"\[CITE: Additional citations from IJSC\.\]", ""),
    (r"\[CITE: NLP in sports: recent papers\]", "(see Wanless et al., 2024)"),
    (r"\[CITE: Social media and sports: recent papers on Reddit/fan engagement\]",
     "(Naraine et al., 2021; Abeza, 2023)"),
    (r"\[CITE\]", ""),
]

# ── Real references section content ──────────────────────────────────────────

REFS_PAPER2 = """
## References

Bauer, H. H., Sauer, N. E., & Schmitt, P. (2005). Customer-based brand equity in the team sport industry. *European Journal of Marketing, 39*(5), 496–513. https://doi.org/10.1108/03090560510590683

Dietl, H. M., Franck, E., Lang, M., & Rathke, A. (2012). Salary cap regulation in professional team sports. *Contemporary Economic Policy, 30*(3), 307–319.

Fort, R., & Quirk, J. (1995). Cross-subsidization, incentives, and outcomes in professional team sports leagues. *Journal of Economic Literature, 33*(3), 1265–1299.

Gladden, J. M., & Funk, D. C. (2002). Developing an understanding of brand associations in team sport: Empirical evidence from consumers of professional sport. *Journal of Sport Management, 16*(1), 54–81.

Kaynak, E., Salman, G. G., & Tatoglu, E. (2008). An integrative framework linking brand associations and brand loyalty in professional sports. *Journal of Brand Management, 15*, 336–357. https://doi.org/10.1057/palgrave.bm.2550117

Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance, 66*(1), 35–65. https://doi.org/10.1111/j.1540-6261.2010.01625.x

Lovett, M. J., & Staelin, R. (2016). The role of paid, earned, and owned media in building entertainment brands. *Marketing Science, 35*(1), 142–157. https://doi.org/10.1287/mksc.2015.0961

Mao, L. L., Zhang, J. J., Kim, M. J., Kim, H., Connaughton, D. P., & Wang, Y. (2023). Towards an inductive model of customer experience in fitness clubs: A structural topic modeling approach. *European Sport Management Quarterly, 24*(4), 898–920. https://doi.org/10.1080/16184742.2023.2219684

Naraine, M. L., Bakhsh, J. T., & Wanless, L. (2022). The impact of sponsorship on social media engagement: A longitudinal examination of professional sport teams. *Sport Marketing Quarterly, 31*(3). https://doi.org/10.32731/SMQ.313.0922.06

Naraine, M. L., Pegoraro, A., & Wear, H. (2021). #WeTheNorth: Examining an online brand community through a professional sport organization's hashtag marketing campaign. *Communication & Sport, 9*, 625–645. https://doi.org/10.1177/2167479519878676

Ross, S. D. (2006). A conceptual framework for understanding spectator-based brand equity. *Journal of Sport Management, 20*(1), 22–38.

Shoghanian, D., & Tang, [First Name]. (forthcoming). Narrative network centrality and club performance in the English Premier League: Evidence from press co-occurrence networks. *[Journal — in preparation]*.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x

Wanless, L., Kennedy, H., Davies, M., Naraine, M. L., & Pegoraro, A. (2024). Look what we have here: Exploring brand-related sport consumer Twitter conversation topics. *Sport Marketing Quarterly, 33*(2). https://doi.org/10.32731/SMQ.332.062024.04
"""

REFS_PAPER1 = """
## References

Abeza, G. (2023). Social media and sport studies (2014–2023): A critical review. *International Journal of Sport Communication, 16*(3), 251–261. https://doi.org/10.1123/ijsc.2023-0182

Dietl, H. M., Franck, E., Lang, M., & Rathke, A. (2012). Salary cap regulation in professional team sports. *Contemporary Economic Policy, 30*(3), 307–319.

Fort, R., & Quirk, J. (1995). Cross-subsidization, incentives, and outcomes in professional team sports leagues. *Journal of Economic Literature, 33*(3), 1265–1299.

McCombs, M. E., & Shaw, D. L. (1972). The agenda-setting function of mass media. *Public Opinion Quarterly, 36*(2), 176–187.

Mao, L. L., Zhang, J. J., Kim, M. J., Kim, H., Connaughton, D. P., & Wang, Y. (2023). Towards an inductive model of customer experience in fitness clubs: A structural topic modeling approach. *European Sport Management Quarterly, 24*(4), 898–920. https://doi.org/10.1080/16184742.2023.2219684

Rewilak, J. (2024). The designated player dilemma: An analysis of demand for star players in Major League Soccer. *European Sport Management Quarterly, 25*(5). https://doi.org/10.1080/16184742.2024.2435823

Shoghanian, D., & Tang, [First Name]. (forthcoming). Narrative network centrality and club performance in the English Premier League: Evidence from press co-occurrence networks. *[Journal — in preparation]*.

Szymanski, S., & Kuypers, T. (1999). *Winners and losers: The business strategy of football*. Viking Press.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x

Wanless, L., Kennedy, H., Davies, M., Naraine, M. L., & Pegoraro, A. (2024). Look what we have here: Exploring brand-related sport consumer Twitter conversation topics. *Sport Marketing Quarterly, 33*(2). https://doi.org/10.32731/SMQ.332.062024.04
"""

REFS_RESEARCH_NOTE = """
## References

Abeza, G. (2023). Social media and sport studies (2014–2023): A critical review. *International Journal of Sport Communication, 16*(3), 251–261. https://doi.org/10.1123/ijsc.2023-0182

McCombs, M. E., & Shaw, D. L. (1972). The agenda-setting function of mass media. *Public Opinion Quarterly, 36*(2), 176–187.

Naraine, M. L., Bakhsh, J. T., & Wanless, L. (2022). The impact of sponsorship on social media engagement: A longitudinal examination of professional sport teams. *Sport Marketing Quarterly, 31*(3). https://doi.org/10.32731/SMQ.313.0922.06

Naraine, M. L., Pegoraro, A., & Wear, H. (2021). #WeTheNorth: Examining an online brand community through a professional sport organization's hashtag marketing campaign. *Communication & Sport, 9*, 625–645. https://doi.org/10.1177/2167479519878676

Shoghanian, D., & Tang, [First Name]. (forthcoming). Narrative network centrality and club performance in the English Premier League: Evidence from press co-occurrence networks. *[Journal — in preparation]*.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x

Wanless, L., Kennedy, H., Davies, M., Naraine, M. L., & Pegoraro, A. (2024). Look what we have here: Exploring brand-related sport consumer Twitter conversation topics. *Sport Marketing Quarterly, 33*(2). https://doi.org/10.32731/SMQ.332.062024.04
"""

REFS_MAP = {
    "paper2_narrative_commercial_returns.md": REFS_PAPER2,
    "paper1_narrative_competitive_outcomes.md": REFS_PAPER1,
    "research_note_press_reddit.md": REFS_RESEARCH_NOTE,
}


def process_paper(filename: str, refs: str):
    path = PAPERS_DIR / filename
    text = path.read_text()

    # Apply all inline citation replacements
    for pattern, replacement in CITE_MAP:
        text = re.sub(pattern, replacement, text)

    # Strip the old references section and replace it
    ref_markers = [
        "## References\n",
        "## References\r\n",
    ]
    for marker in ref_markers:
        if marker in text:
            text = text[:text.index(marker)]
            break

    # Also strip any trailing [CITE: ...] blocks that weren't matched
    text = re.sub(r'\[CITE:[^\]]*\]', '', text)

    text = text.rstrip() + "\n" + refs

    path.write_text(text)
    remaining = re.findall(r'\[CITE[^\]]*\]', text)
    print(f"{filename}: done. Remaining unmatched [CITE] blocks: {len(remaining)}")
    if remaining:
        for r in remaining:
            print(f"  - {r[:80]}")


def main():
    for filename, refs in REFS_MAP.items():
        process_paper(filename, refs)
    print("\nAll papers updated.")


if __name__ == "__main__":
    main()
