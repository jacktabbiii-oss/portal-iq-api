"""
Configuration Management

Centralized configuration for the ML Engine.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class Config:
    """Central configuration manager."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "config.yaml").exists():
                return current
            current = current.parent
        return Path(__file__).parent.parent.parent

    def _load_config(self) -> None:
        """Load configuration from file and environment."""
        load_dotenv()

        root = self._find_project_root()
        config_path = root / "config.yaml"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

        # Set defaults
        self._config.setdefault("data_dir", str(root / "data"))
        self._config.setdefault("cache_hours", 24)
        self._config.setdefault("rate_limit_seconds", 0.5)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_env(self, key: str, default: str = "") -> str:
        """Get an environment variable."""
        return os.getenv(key, default)

    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return Path(self.get("data_dir", "data"))

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        cache = self.data_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    @property
    def cfbd_api_key(self) -> str:
        """Get CFBD API key."""
        return self.get_env("CFBD_API_KEY", "")

    @property
    def pocketbase_url(self) -> str:
        """Get PocketBase URL."""
        return self.get_env("POCKETBASE_URL", "http://localhost:8090")

    @property
    def pocketbase_secret(self) -> str:
        """Get PocketBase JWT secret."""
        return self.get_env("POCKETBASE_JWT_SECRET", "")

    # School tiers for NIL valuation
    @property
    def school_tiers(self) -> Dict[str, list]:
        """Get school tier classifications."""
        return self.get("school_tiers", {
            "blue_blood": [
                "Alabama", "Ohio State", "Georgia", "Clemson",
                "Oklahoma", "Notre Dame", "Texas", "USC",
            ],
            "elite": [
                "Michigan", "Penn State", "LSU", "Florida",
                "Oregon", "Tennessee", "Miami", "Texas A&M",
            ],
            "power_brand": [
                "Auburn", "Florida State", "Wisconsin", "UCLA",
                "Nebraska", "Arkansas", "South Carolina", "Ole Miss",
            ],
            "p4_mid": [
                "Iowa", "Michigan State", "NC State", "Virginia Tech",
                "Louisville", "Pittsburgh", "West Virginia", "Arizona State",
            ],
            "g5_strong": [
                "Boise State", "Memphis", "UCF", "SMU",
                "Tulane", "Liberty", "James Madison", "Appalachian State",
            ],
            "g5": [],  # All other G5 schools
        })

    # NIL value tiers
    @property
    def nil_tiers(self) -> Dict[str, Dict[str, Any]]:
        """Get NIL tier thresholds."""
        return self.get("nil_tiers", {
            "mega": {"min_value": 1_000_000, "label": "Mega Deal"},
            "premium": {"min_value": 500_000, "label": "Premium"},
            "solid": {"min_value": 100_000, "label": "Solid"},
            "moderate": {"min_value": 25_000, "label": "Moderate"},
            "entry": {"min_value": 0, "label": "Entry Level"},
        })

    # Position values for NFL contracts
    @property
    def position_values(self) -> Dict[str, float]:
        """Get position value multipliers for NFL contracts."""
        return self.get("position_values", {
            "QB": 1.0,
            "EDGE": 0.85,
            "WR": 0.80,
            "CB": 0.75,
            "OT": 0.75,
            "DT": 0.65,
            "S": 0.60,
            "LB": 0.55,
            "TE": 0.55,
            "IOL": 0.50,
            "RB": 0.45,
            "K": 0.20,
            "P": 0.15,
            "LS": 0.10,
        })


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
