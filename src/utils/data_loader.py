"""
Data Loader Utilities

Unified data loading for ML Engine with caching support.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import get_config

logger = logging.getLogger(__name__)


class DataLoader:
    """Unified data loader for ML Engine."""

    def __init__(self):
        self.config = get_config()
        self.data_dir = self.config.data_dir
        self.cache_dir = self.config.cache_dir

    def load_csv(
        self,
        filename: str,
        subdir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV file from the data directory.

        Args:
            filename: Name of the CSV file
            subdir: Optional subdirectory within data_dir

        Returns:
            DataFrame with the loaded data
        """
        if subdir:
            path = self.data_dir / subdir / filename
        else:
            path = self.data_dir / filename

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()

        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return pd.DataFrame()

    def load_parquet(
        self,
        filename: str,
        subdir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a Parquet file from the data directory.

        Args:
            filename: Name of the Parquet file
            subdir: Optional subdirectory within data_dir

        Returns:
            DataFrame with the loaded data
        """
        if subdir:
            path = self.data_dir / subdir / filename
        else:
            path = self.data_dir / filename

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()

        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return pd.DataFrame()

    def save_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        subdir: Optional[str] = None,
    ) -> bool:
        """
        Save a DataFrame to CSV.

        Args:
            df: DataFrame to save
            filename: Name of the output file
            subdir: Optional subdirectory within data_dir

        Returns:
            True if successful, False otherwise
        """
        if subdir:
            path = self.data_dir / subdir
        else:
            path = self.data_dir

        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename

        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            return False

    def save_parquet(
        self,
        df: pd.DataFrame,
        filename: str,
        subdir: Optional[str] = None,
    ) -> bool:
        """
        Save a DataFrame to Parquet.

        Args:
            df: DataFrame to save
            filename: Name of the output file
            subdir: Optional subdirectory within data_dir

        Returns:
            True if successful, False otherwise
        """
        if subdir:
            path = self.data_dir / subdir
        else:
            path = self.data_dir

        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename

        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            return False

    def load_processed_data(self, product: str) -> Dict[str, pd.DataFrame]:
        """
        Load all processed data for a product.

        Args:
            product: 'portal_iq' or 'cap_iq'

        Returns:
            Dictionary of DataFrames
        """
        processed_dir = self.data_dir / "processed" / product

        if not processed_dir.exists():
            logger.warning(f"Processed directory not found: {processed_dir}")
            return {}

        data = {}
        for file in processed_dir.glob("*.parquet"):
            name = file.stem
            data[name] = pd.read_parquet(file)
            logger.info(f"Loaded {name}: {len(data[name])} rows")

        return data

    def get_available_years(self, data_type: str) -> List[int]:
        """
        Get available years for a data type.

        Args:
            data_type: Type of data (e.g., 'cfb_stats', 'nfl_contracts')

        Returns:
            List of available years
        """
        cache_pattern = f"*{data_type}*.csv"
        files = list(self.cache_dir.glob(cache_pattern))

        years = set()
        for f in files:
            # Extract years from filename patterns like "cfb_stats_2020_2024_cache.csv"
            parts = f.stem.split("_")
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    years.add(int(part))

        return sorted(years)

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cached files.

        Args:
            pattern: Optional glob pattern to match specific files

        Returns:
            Number of files deleted
        """
        if pattern:
            files = list(self.cache_dir.glob(pattern))
        else:
            files = list(self.cache_dir.glob("*"))

        deleted = 0
        for f in files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Could not delete {f}: {e}")

        logger.info(f"Cleared {deleted} cache files")
        return deleted
