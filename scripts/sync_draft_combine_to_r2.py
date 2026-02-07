"""
Sync Draft and Combine Data to R2

Uploads NFL draft data (2020-2025) and combine data to Cloudflare R2 storage.
Cleans and normalizes data before upload.

Usage:
    python scripts/sync_draft_combine_to_r2.py

Requires environment variables:
    - R2_ENDPOINT_URL
    - R2_ACCESS_KEY_ID
    - R2_SECRET_ACCESS_KEY
    - R2_BUCKET_NAME (optional, defaults to 'portal-iq-data')
"""

import os
import sys
from pathlib import Path
import re

import pandas as pd
import boto3
from botocore.config import Config
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_file = project_root / "ml-engine" / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded environment from {env_file}")
else:
    # Try parent directory
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")


def get_s3_client():
    """Get configured S3 client for R2."""
    return boto3.client(
        's3',
        endpoint_url=os.getenv('R2_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
        region_name='auto',
        config=Config(signature_version='s3v4')
    )


def clean_string(s):
    """Clean string by stripping whitespace."""
    if pd.isna(s):
        return None
    return str(s).strip()


def clean_currency(s):
    """Convert currency string to float."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    s = re.sub(r'[$,]', '', s)
    s = re.sub(r'%', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def process_draft_file(filepath):
    """Process a single NFL draft CSV file.

    Args:
        filepath: Path to the draft CSV file

    Returns:
        DataFrame with cleaned draft data
    """
    print(f"Processing: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath, encoding='utf-8')

    # Clean column names
    df.columns = [clean_string(c) for c in df.columns]

    # Extract year from filename
    filename = os.path.basename(filepath)
    year_match = re.search(r'(\d{4})', filename)
    year = int(year_match.group(1)) if year_match else 2025

    # Rename columns to standard format
    column_map = {
        'Round': 'round',
        'Pick': 'pick',
        'Team': 'team',
        'Player': 'name',
        'Pos': 'position',
        'Age': 'age',
        'Pre-Draft Team': 'college',
        'Yrs': 'contract_years',
        'Value': 'contract_value',
        'AAV': 'aav',
        'Signing Bonus': 'signing_bonus',
        'Guaranteed': 'guaranteed',
        'Guaranteed %': 'guaranteed_pct',
        'Agent': 'agent',
    }

    # Rename columns that exist
    for old_name, new_name in column_map.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Clean string columns
    string_cols = ['name', 'team', 'position', 'college', 'agent']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_string)

    # Clean numeric columns
    numeric_cols = ['round', 'pick', 'age', 'contract_years']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].apply(clean_string), errors='coerce')

    # Clean currency columns
    currency_cols = ['contract_value', 'aav', 'signing_bonus', 'guaranteed', 'guaranteed_pct']
    for col in currency_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)

    # Add year column
    df['year'] = year

    # Calculate overall pick number
    if 'round' in df.columns and 'pick' in df.columns:
        # Calculate overall based on 32 picks per round (approximate)
        # Use Int64 to handle NaN values
        df['overall'] = ((df['round'] - 1) * 32 + df['pick']).astype('Int64')

    # Drop any empty rows
    df = df.dropna(subset=['name'])

    print(f"  Loaded {len(df)} picks from {year}")
    return df


def process_combine_file(filepath):
    """Process the combine data CSV file.

    Args:
        filepath: Path to the combine CSV file

    Returns:
        DataFrame with cleaned combine data
    """
    print(f"Processing: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath, encoding='utf-8')

    # Rename columns to standard format
    column_map = {
        'Year': 'combine_year',
        'Player': 'name',
        'Position': 'position',
        'School': 'school',
        '40 Yard Dash': 'forty',
        'Vertical Jump': 'vertical',
        'Broad Jump': 'broad_jump',
        '20  Yard Shuttle': 'shuttle',
        '20 Yard Shuttle': 'shuttle',
        'Bench Press': 'bench',
    }

    for old_name, new_name in column_map.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Clean string columns
    string_cols = ['name', 'position', 'school']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_string)

    # Ensure numeric columns
    numeric_cols = ['combine_year', 'forty', 'vertical', 'broad_jump', 'shuttle', 'bench']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any empty rows
    df = df.dropna(subset=['name'])

    print(f"  Loaded {len(df)} combine records")
    return df


def upload_to_r2(df, key, bucket=None):
    """Upload DataFrame to R2 as CSV.

    Args:
        df: DataFrame to upload
        key: S3 key (path in bucket)
        bucket: Bucket name (defaults to R2_BUCKET_NAME env var)
    """
    if bucket is None:
        bucket = os.getenv('R2_BUCKET_NAME', 'portal-iq-data')

    client = get_s3_client()

    # Convert to CSV bytes
    csv_buffer = df.to_csv(index=False)

    # Upload
    print(f"Uploading to s3://{bucket}/{key}")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.encode('utf-8'),
        ContentType='text/csv'
    )
    print(f"  Uploaded {len(df)} records")


def main():
    """Main function to sync all draft and combine data."""

    # Check for R2 credentials
    if not os.getenv('R2_ENDPOINT_URL'):
        print("ERROR: R2_ENDPOINT_URL not set")
        print("Set R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(1)

    downloads_path = Path("c:/Users/kerra/Downloads")

    # ==========================================================================
    # Process Draft Data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PROCESSING NFL DRAFT DATA (2020-2025)")
    print("=" * 60)

    all_draft_data = []

    for year in range(2020, 2026):
        draft_file = downloads_path / f"NFL DRAFT {year}.csv"
        if draft_file.exists():
            df = process_draft_file(draft_file)
            all_draft_data.append(df)
        else:
            print(f"  NOT FOUND: {draft_file}")

    if all_draft_data:
        combined_draft = pd.concat(all_draft_data, ignore_index=True)
        print(f"\nTotal draft picks: {len(combined_draft)}")

        # Upload combined file
        upload_to_r2(combined_draft, "data/nfl_draft_picks.csv")

        # Also upload individual years
        for year in combined_draft['year'].unique():
            year_df = combined_draft[combined_draft['year'] == year]
            upload_to_r2(year_df, f"draft/{year}_nfl_draft.csv")

    # ==========================================================================
    # Process Combine Data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PROCESSING COMBINE DATA")
    print("=" * 60)

    combine_file = downloads_path / "Historical Combine data for Modeling - Overall.csv"
    if combine_file.exists():
        combine_df = process_combine_file(combine_file)
        print(f"\nTotal combine records: {len(combine_df)}")

        # Upload combined file
        upload_to_r2(combine_df, "combine/historical_combine_data.csv")
        upload_to_r2(combine_df, "processed/combine_data.csv")  # Also in processed/

        # Upload by year
        for year in combine_df['combine_year'].unique():
            if pd.notna(year):
                year_df = combine_df[combine_df['combine_year'] == year]
                upload_to_r2(year_df, f"combine/{int(year)}_combine.csv")
    else:
        print(f"  NOT FOUND: {combine_file}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)

    if all_draft_data:
        print(f"Draft data: {len(combined_draft)} picks ({len(all_draft_data)} years)")
        print(f"  Years: {sorted(combined_draft['year'].unique().tolist())}")

    if combine_file.exists():
        print(f"Combine data: {len(combine_df)} records")
        years = sorted([int(y) for y in combine_df['combine_year'].unique() if pd.notna(y)])
        print(f"  Years: {years}")


if __name__ == "__main__":
    main()
