"""
College Football Data Collectors

Data collection modules for Portal IQ:
- CFB stats and game data (player stats, team data, game results)
- Transfer portal data (entries, enrichment, outcomes)
- Recruiting rankings (player rankings, team classes, performance analysis)
- NIL valuations (valuations, social media, collective budgets)
- Draft history (draft picks, combine, college-to-NFL outcomes)
- On3 scraper (NIL rankings, player profiles)
- NIL data integrator (merge NIL data with performance stats)
"""

# Lazy imports to avoid circular import issues
def __getattr__(name):
    if name == "CFBStatsCollector":
        from .cfb_stats import CFBStatsCollector
        return CFBStatsCollector
    elif name == "CFBPortalCollector":
        from .cfb_portal import CFBPortalCollector
        return CFBPortalCollector
    elif name == "CFBRecruitingCollector":
        from .cfb_recruiting import CFBRecruitingCollector
        return CFBRecruitingCollector
    elif name == "CFBNILCollector":
        from .cfb_nil import CFBNILCollector
        return CFBNILCollector
    elif name == "DraftHistoryCollector":
        from .draft_history import DraftHistoryCollector
        return DraftHistoryCollector
    elif name == "On3Scraper":
        from .on3_scraper import On3Scraper
        return On3Scraper
    elif name == "NILDataIntegrator":
        from .nil_data_integrator import NILDataIntegrator
        return NILDataIntegrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CFBStatsCollector",
    "CFBPortalCollector",
    "CFBRecruitingCollector",
    "CFBNILCollector",
    "DraftHistoryCollector",
    "On3Scraper",
    "NILDataIntegrator",
]
