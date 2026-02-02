"""
Data Collection Module

Collectors for college football (Portal IQ) and NFL (Cap IQ) data.
Provides unified interfaces for fetching player statistics, recruiting data,
portal entries, NIL valuations, and NFL contract information.
"""

from .college.cfb_stats import CFBStatsCollector
from .college.cfb_recruiting import CFBRecruitingCollector
from .college.cfb_portal import CFBPortalCollector
from .college.cfb_nil import CFBNILCollector
from .nfl.nfl_stats import NFLStatsCollector
from .nfl.nfl_contracts import NFLContractCollector

__all__ = [
    # College Football Collectors (Portal IQ)
    "CFBStatsCollector",
    "CFBRecruitingCollector",
    "CFBPortalCollector",
    "CFBNILCollector",
    # NFL Collectors (Cap IQ)
    "NFLStatsCollector",
    "NFLContractCollector",
]
