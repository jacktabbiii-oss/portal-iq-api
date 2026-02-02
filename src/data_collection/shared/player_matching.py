"""
Player Matching Utilities

Fuzzy matching and entity resolution for player names across data sources.
Used to link college players to NFL players, match across different data sources, etc.
"""

import re
from typing import Optional, List, Dict, Tuple
from difflib import SequenceMatcher

import pandas as pd


class PlayerMatcher:
    """Matches player names across different data sources."""

    def __init__(self):
        self._cache: Dict[str, str] = {}

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize a player name for matching."""
        if not name:
            return ""

        name = name.lower()

        # Remove suffixes
        for suffix in [" jr", " jr.", " sr", " sr.", " ii", " iii", " iv", " v"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        # Remove punctuation
        name = re.sub(r"[^\w\s]", "", name)
        name = " ".join(name.split())

        return name

    @staticmethod
    def similarity(name1: str, name2: str) -> float:
        """Calculate similarity between two names."""
        n1 = PlayerMatcher.normalize_name(name1)
        n2 = PlayerMatcher.normalize_name(name2)

        if n1 == n2:
            return 1.0

        return SequenceMatcher(None, n1, n2).ratio()

    def find_best_match(
        self,
        name: str,
        candidates: List[str],
        threshold: float = 0.8,
    ) -> Optional[Tuple[str, float]]:
        """Find best matching name from candidates."""
        norm = self.normalize_name(name)

        if norm in self._cache and self._cache[norm] in candidates:
            return (self._cache[norm], 1.0)

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = self.similarity(name, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= threshold and best_match:
            self._cache[norm] = best_match
            return (best_match, best_score)

        return None

    def match_dataframes(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        col1: str = "player_name",
        col2: str = "player_name",
        threshold: float = 0.8,
    ) -> pd.DataFrame:
        """Match players between two DataFrames."""
        matches = []

        for idx, row in df1.iterrows():
            result = self.find_best_match(
                row[col1],
                df2[col2].tolist(),
                threshold
            )
            if result:
                matched_name, score = result
                matches.append({
                    "df1_idx": idx,
                    "df1_name": row[col1],
                    "df2_name": matched_name,
                    "score": score,
                })

        return pd.DataFrame(matches)

    def create_player_id(
        self,
        name: str,
        team: Optional[str] = None,
        position: Optional[str] = None,
    ) -> str:
        """Create unique player ID."""
        parts = [self.normalize_name(name).replace(" ", "_")]
        if team:
            parts.append(self.normalize_name(team)[:10].replace(" ", ""))
        if position:
            parts.append(position.lower())
        return "_".join(parts)

    @staticmethod
    def parse_name(full_name: str) -> Dict[str, str]:
        """Parse full name into components."""
        parts = full_name.split()
        if not parts:
            return {"first": "", "last": "", "suffix": ""}

        suffix = ""
        suffixes = ["Jr", "Jr.", "Sr", "Sr.", "II", "III", "IV", "V"]
        if parts[-1] in suffixes:
            suffix = parts[-1]
            parts = parts[:-1]

        if len(parts) == 1:
            return {"first": parts[0], "last": "", "suffix": suffix}

        return {"first": parts[0], "last": " ".join(parts[1:]), "suffix": suffix}
