"""
Data Collection Module

Collectors for college football (Portal IQ) and NFL (Cap IQ) data.
Provides unified interfaces for fetching player statistics, recruiting data,
portal entries, NIL valuations, and NFL contract information.
"""

# Lazy imports to avoid import errors when some modules aren't needed
def __getattr__(name):
    if name == "CFBStatsCollector":
        from .college.cfb_stats import CFBStatsCollector
        return CFBStatsCollector
    elif name == "CFBRecruitingCollector":
        from .college.cfb_recruiting import CFBRecruitingCollector
        return CFBRecruitingCollector
    elif name == "CFBPortalCollector":
        from .college.cfb_portal import CFBPortalCollector
        return CFBPortalCollector
    elif name == "CFBNILCollector":
        from .college.cfb_nil import CFBNILCollector
        return CFBNILCollector
    elif name == "On3Scraper":
        from .college.on3_scraper import On3Scraper
        return On3Scraper
    elif name == "NILDataIntegrator":
        from .college.nil_data_integrator import NILDataIntegrator
        return NILDataIntegrator
    elif name == "NFLStatsCollector":
        from .nfl.player_stats import NFLPlayerStatsCollector as NFLStatsCollector
        return NFLStatsCollector
    elif name == "NFLContractCollector":
        from .nfl.contracts import NFLContractCollector
        return NFLContractCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # College Football Collectors (Portal IQ)
    "CFBStatsCollector",
    "CFBRecruitingCollector",
    "CFBPortalCollector",
    "CFBNILCollector",
    "On3Scraper",
    "NILDataIntegrator",
    # NFL Collectors (Cap IQ)
    "NFLStatsCollector",
    "NFLContractCollector",
]
