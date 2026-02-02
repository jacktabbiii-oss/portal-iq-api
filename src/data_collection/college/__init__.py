"""
College Football Data Collectors

Data collection modules for Portal IQ:
- CFB stats and game data (player stats, team data, game results)
- Transfer portal data (entries, enrichment, outcomes)
- Recruiting rankings (player rankings, team classes, performance analysis)
- NIL valuations (valuations, social media, collective budgets)
- Draft history (draft picks, combine, college-to-NFL outcomes)
"""

from .cfb_stats import CFBStatsCollector
from .cfb_portal import CFBPortalCollector
from .cfb_recruiting import CFBRecruitingCollector
from .cfb_nil import CFBNILCollector
from .draft_history import DraftHistoryCollector

__all__ = [
    "CFBStatsCollector",
    "CFBPortalCollector",
    "CFBRecruitingCollector",
    "CFBNILCollector",
    "DraftHistoryCollector",
]
