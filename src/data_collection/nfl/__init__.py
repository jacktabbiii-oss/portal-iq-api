"""
NFL Data Collectors

Data collection modules for Cap IQ:
- Contract data
- Salary cap data
- Player stats and values
- Injury history
"""

from .contracts import ContractDataCollector
from .cap_data import CapDataCollector
from .player_stats import NFLPlayerStatsCollector

__all__ = [
    "ContractDataCollector",
    "CapDataCollector",
    "NFLPlayerStatsCollector",
]
