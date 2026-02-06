"""S3/R2 Storage Client for FastAPI ML Engine.

Handles loading data files from Cloudflare R2 (S3-compatible) storage.
Falls back to local files during development.

This is the FastAPI version - no Streamlit dependencies.
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import boto3 (optional dependency)
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed - S3/R2 storage disabled, using local files")


# =============================================================================
# Configuration
# =============================================================================

def get_s3_config() -> Dict[str, str]:
    """Get S3/R2 configuration from environment."""
    return {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL", ""),
        "access_key_id": os.getenv("R2_ACCESS_KEY_ID", ""),
        "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        "bucket_name": os.getenv("R2_BUCKET_NAME", "portal-iq-data"),
        "region": os.getenv("R2_REGION", "auto"),
    }


def is_s3_configured() -> bool:
    """Check if S3/R2 is properly configured."""
    if not BOTO3_AVAILABLE:
        return False

    config = get_s3_config()
    required = ["endpoint_url", "access_key_id", "secret_access_key"]
    return all(config.get(key) for key in required)


# =============================================================================
# S3 Client
# =============================================================================

class S3StorageClient:
    """Client for reading data from S3/R2 storage."""

    def __init__(self):
        """Initialize the S3 client."""
        self.config = get_s3_config()
        self.bucket = self.config["bucket_name"]
        self._client = None
        self._cache_dir = Path.home() / ".cache" / "portal-iq" / "data"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        """Lazy-load S3 client."""
        if self._client is None and BOTO3_AVAILABLE and is_s3_configured():
            try:
                self._client = boto3.client(
                    "s3",
                    endpoint_url=self.config["endpoint_url"],
                    aws_access_key_id=self.config["access_key_id"],
                    aws_secret_access_key=self.config["secret_access_key"],
                    region_name=self.config["region"],
                    config=Config(
                        signature_version="s3v4",
                        retries={"max_attempts": 3, "mode": "adaptive"},
                    ),
                )
                logger.info("S3/R2 client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self._client = None
        return self._client

    def _get_cache_path(self, key: str) -> Path:
        """Get local cache path for an S3 key."""
        safe_name = key.replace("/", "_").replace("\\", "_")
        return self._cache_dir / safe_name

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 1) -> bool:
        """Check if cached file is still valid."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(hours=max_age_hours)

    def download_file(
        self,
        key: str,
        local_path: Optional[Path] = None,
        use_cache: bool = True,
        cache_hours: int = 1,
    ) -> Optional[Path]:
        """Download a file from S3/R2.

        Args:
            key: S3 object key (e.g., "processed/portal_nil_valuations.csv")
            local_path: Optional local path to save to
            use_cache: Whether to use local cache
            cache_hours: How long to cache files locally

        Returns:
            Path to local file, or None if download failed
        """
        if not self.client:
            logger.debug(f"S3 not available, cannot download {key}")
            return None

        if local_path is None:
            local_path = self._get_cache_path(key)

        if use_cache and self._is_cache_valid(local_path, cache_hours):
            logger.debug(f"Using cached file: {local_path}")
            return local_path

        try:
            logger.info(f"Downloading from S3: {key}")
            self.client.download_file(self.bucket, key, str(local_path))
            return local_path

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404" or error_code == "NoSuchKey":
                logger.warning(f"S3 file not found: {key}")
            else:
                logger.error(f"S3 download failed: {key} - {e}")
            return None

        except Exception as e:
            logger.error(f"S3 download failed: {key} - {e}")
            return None

    def read_csv(
        self,
        key: str,
        use_cache: bool = True,
        cache_hours: int = 1,
        **pandas_kwargs,
    ) -> pd.DataFrame:
        """Read a CSV file from S3/R2.

        Args:
            key: S3 object key
            use_cache: Whether to use local cache
            cache_hours: Cache duration
            **pandas_kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame, or empty DataFrame if read failed
        """
        local_path = self.download_file(key, use_cache=use_cache, cache_hours=cache_hours)

        if local_path and local_path.exists():
            try:
                df = pd.read_csv(local_path, **pandas_kwargs)
                logger.info(f"Loaded {key}: {len(df)} rows")
                return df
            except Exception as e:
                logger.error(f"Failed to parse CSV: {key} - {e}")

        return pd.DataFrame()

    def read_csv_direct(self, key: str, **pandas_kwargs) -> pd.DataFrame:
        """Read CSV directly from S3 without caching."""
        if not self.client:
            return pd.DataFrame()

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()), **pandas_kwargs)
            logger.info(f"Direct load {key}: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to read CSV directly: {key} - {e}")
            return pd.DataFrame()

    def file_exists(self, key: str) -> bool:
        """Check if a file exists in S3/R2."""
        if not self.client:
            return False

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
        except Exception:
            return False

    def list_files(self, prefix: str = "") -> list:
        """List files in the bucket with optional prefix."""
        if not self.client:
            return []

        try:
            response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            files = [obj["Key"] for obj in response.get("Contents", [])]
            logger.debug(f"Listed {len(files)} files with prefix '{prefix}'")
            return files

        except Exception as e:
            logger.error(f"Failed to list S3 files: {prefix} - {e}")
            return []


# =============================================================================
# Singleton & Helper Functions
# =============================================================================

_s3_client: Optional[S3StorageClient] = None


def get_s3_client() -> S3StorageClient:
    """Get singleton S3 client instance."""
    global _s3_client
    if _s3_client is None:
        _s3_client = S3StorageClient()
    return _s3_client


def load_csv_with_fallback(
    s3_key: str,
    local_path: Optional[Path],
    cache_hours: int = 1,
    **pandas_kwargs,
) -> pd.DataFrame:
    """Load CSV from S3, falling back to local file.

    This is the main function to use throughout the API.

    Args:
        s3_key: S3 object key (e.g., "processed/portal_nil_valuations.csv")
        local_path: Local fallback path (can be None for S3-only)
        cache_hours: S3 cache duration
        **pandas_kwargs: Arguments for pd.read_csv

    Returns:
        DataFrame
    """
    # Try S3 first if configured
    if is_s3_configured():
        client = get_s3_client()
        df = client.read_csv(s3_key, cache_hours=cache_hours, **pandas_kwargs)
        if not df.empty:
            return df
        logger.debug(f"S3 read failed for {s3_key}, trying local")

    # Fallback to local file
    if local_path and local_path.exists():
        try:
            df = pd.read_csv(local_path, **pandas_kwargs)
            logger.info(f"Local load {local_path}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to read local CSV: {local_path} - {e}")

    logger.warning(f"No data available for {s3_key}")
    return pd.DataFrame()


# =============================================================================
# Data Path Mapping
# =============================================================================

DATA_PATHS = {
    # NIL & Portal Data
    "portal_nil_valuations": "processed/portal_nil_valuations.csv",
    "on3_transfer_portal": "processed/on3_transfer_portal.csv",
    "on3_all_nil_rankings": "processed/on3_all_nil_rankings.csv",
    "on3_team_portal_rankings": "processed/on3_team_portal_rankings.csv",

    # CFBD Data
    "cfbd_rosters": "processed/cfbd_rosters.csv",
    "cfbd_player_stats": "processed/cfbd_player_stats.csv",
    "cfbd_team_talent": "processed/cfbd_team_talent.csv",
    "cfbd_sp_ratings": "processed/cfbd_sp_ratings.csv",

    # PFF Grades
    "pff_player_grades": "processed/pff_player_grades.csv",

    # ESPN Data
    "espn_rosters": "processed/espn_rosters.csv",
}


def load_data(name: str, cache_hours: int = 1, **kwargs) -> pd.DataFrame:
    """Load a named dataset from S3 or local storage.

    Args:
        name: Dataset name (e.g., "portal_nil_valuations")
        cache_hours: How long to cache S3 data
        **kwargs: Additional pd.read_csv arguments

    Returns:
        DataFrame
    """
    if name not in DATA_PATHS:
        logger.error(f"Unknown dataset: {name}")
        return pd.DataFrame()

    s3_key = DATA_PATHS[name]

    # Build local path relative to ml-engine data directory
    project_root = Path(__file__).parent.parent.parent  # ml-engine/src/utils -> ml-engine
    local_path = project_root / "data" / s3_key.split("/", 1)[-1]

    return load_csv_with_fallback(s3_key, local_path, cache_hours, **kwargs)
