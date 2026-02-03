"""
Generate NIL Valuations from CFBD Data

Loads player stats, pivots to wide format, merges with recruiting data,
and runs through the custom NIL valuator to produce performance-based valuations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from models.custom_nil_valuator import CustomNILValuator


def load_and_pivot_stats(data_dir: Path, season: int = 2024) -> pd.DataFrame:
    """
    Load player stats and pivot from long to wide format.

    CFBD data is in format: player, category, statType, stat
    We need: player, passing_yards, passing_tds, rushing_yards, etc.
    """
    print(f"Loading player stats for {season} season...")
    stats_file = data_dir / "raw" / "player_season_stats.csv"
    df = pd.read_csv(stats_file)

    # Filter to requested season
    df = df[df['season'] == season]
    print(f"  Found {len(df):,} stat records for {season}")

    # Create a combined stat key: category_statType
    df['stat_key'] = df['category'] + '_' + df['statType']

    # Pivot to wide format
    pivot_df = df.pivot_table(
        index=['player', 'playerId', 'team', 'conference', 'position'],
        columns='stat_key',
        values='stat',
        aggfunc='first'  # In case of duplicates
    ).reset_index()

    # Flatten column names
    pivot_df.columns = [col if isinstance(col, str) else col for col in pivot_df.columns]

    print(f"  Pivoted to {len(pivot_df):,} unique players")
    return pivot_df


def load_recruiting_data(data_dir: Path) -> pd.DataFrame:
    """Load recruiting rankings for star ratings."""
    print("Loading recruiting data...")
    recruiting_file = data_dir / "raw" / "recruiting_rankings.csv"
    df = pd.read_csv(recruiting_file)

    # Get most recent recruiting year per player
    df = df.sort_values('year', ascending=False)
    df = df.drop_duplicates(subset=['name'], keep='first')

    # Select relevant columns
    cols = ['name', 'stars', 'rating', 'ranking']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()

    # Rename for consistency
    rename_map = {
        'name': 'recruit_name',
        'stars': 'recruiting_stars',
        'rating': 'recruiting_rating',
        'ranking': 'national_rank'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    print(f"  Loaded {len(df):,} recruiting records")
    return df


def map_stats_to_valuator(df: pd.DataFrame) -> pd.DataFrame:
    """Map CFBD stat columns to valuator expected columns."""

    # Create mapping from CFBD columns to valuator inputs
    column_mapping = {
        'player': 'player_name',
        'team': 'school',
        'position': 'position',
        'conference': 'conference',

        # Passing stats
        'passing_YDS': 'passing_yards',
        'passing_TD': 'passing_tds',

        # Rushing stats
        'rushing_YDS': 'rushing_yards',
        'rushing_TD': 'rushing_tds',
        'rushing_CAR': 'rushing_carries',

        # Receiving stats
        'receiving_YDS': 'receiving_yards',
        'receiving_TD': 'receiving_tds',
        'receiving_REC': 'receptions',

        # Defensive stats
        'defensive_TOT': 'tackles',
        'defensive_SACKS': 'sacks',
        'defensive_TFL': 'tackles_for_loss',
        'defensive_PD': 'passes_defended',

        # Turnovers
        'interceptions_INT': 'interceptions',
        'fumbles_FUM': 'fumbles',
    }

    # Rename columns that exist
    rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_cols)

    # Fill NaN with 0 for numeric stats
    stat_cols = [
        'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
        'receiving_yards', 'receiving_tds', 'tackles', 'sacks', 'interceptions'
    ]
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


def merge_recruiting(stats_df: pd.DataFrame, recruiting_df: pd.DataFrame) -> pd.DataFrame:
    """Merge recruiting data with stats using fuzzy name matching."""

    # Try exact name match first
    merged = stats_df.merge(
        recruiting_df,
        left_on='player_name',
        right_on='recruit_name',
        how='left'
    )

    # Fill missing recruiting data with defaults
    merged['recruiting_stars'] = merged['recruiting_stars'].fillna(0).astype(int)
    merged['national_rank'] = merged['national_rank'].fillna(9999)

    matched = merged['recruit_name'].notna().sum()
    print(f"  Matched {matched:,} players with recruiting data")

    return merged


def run_valuations(df: pd.DataFrame) -> pd.DataFrame:
    """Run the NIL valuator on all players."""
    print("Running NIL valuations...")

    valuator = CustomNILValuator(calibration_factor=1.0)

    # Run valuator
    result_df = valuator.valuate_dataframe(df)

    # Sort by valuation
    result_df = result_df.sort_values('custom_nil_value', ascending=False)

    print(f"  Generated valuations for {len(result_df):,} players")
    return result_df


def main():
    print("=" * 60)
    print("NIL VALUATION GENERATOR")
    print("Performance-based valuations from CFBD data")
    print("=" * 60 + "\n")

    data_dir = Path(__file__).parent / "data"

    # Check for available seasons
    stats_file = data_dir / "raw" / "player_season_stats.csv"
    all_stats = pd.read_csv(stats_file, usecols=['season'])
    seasons = sorted(all_stats['season'].unique(), reverse=True)
    print(f"Available seasons: {seasons}")

    # Use most recent season
    season = seasons[0] if seasons else 2024
    print(f"Using season: {season}\n")

    # 1. Load and pivot stats
    stats_df = load_and_pivot_stats(data_dir, season=season)

    # 2. Map columns
    stats_df = map_stats_to_valuator(stats_df)

    # 3. Load and merge recruiting data
    recruiting_df = load_recruiting_data(data_dir)
    merged_df = merge_recruiting(stats_df, recruiting_df)

    # 4. Run valuations
    valuations_df = run_valuations(merged_df)

    # 5. Select output columns
    output_cols = [
        'player_name', 'position', 'school', 'conference',
        'custom_nil_value', 'nil_tier', 'valuation_confidence',
        'performance_value', 'market_value', 'potential_value',
        'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
        'receiving_yards', 'receiving_tds', 'tackles', 'sacks', 'interceptions',
        'recruiting_stars', 'national_rank'
    ]
    available_cols = [c for c in output_cols if c in valuations_df.columns]
    output_df = valuations_df[available_cols].copy()

    # 6. Save results
    output_path = data_dir / "processed" / f"nil_valuations_{season}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # 7. Print top players
    print("\n" + "=" * 80)
    print(f"TOP 25 NIL VALUATIONS - {season} SEASON (Performance-Based)")
    print("=" * 80)

    top_25 = output_df.head(25)
    for i, row in top_25.iterrows():
        val = row['custom_nil_value']
        name = row['player_name'][:25]
        pos = row['position']
        school = row['school'][:15] if 'school' in row else 'Unknown'
        tier = row['nil_tier']

        # Key stats based on position
        if pos == 'QB':
            stats = f"Pass: {int(row.get('passing_yards', 0)):,}yds, {int(row.get('passing_tds', 0))}TD"
        elif pos in ['RB', 'FB']:
            stats = f"Rush: {int(row.get('rushing_yards', 0)):,}yds, {int(row.get('rushing_tds', 0))}TD"
        elif pos in ['WR', 'TE']:
            stats = f"Rec: {int(row.get('receiving_yards', 0)):,}yds, {int(row.get('receiving_tds', 0))}TD"
        else:
            stats = f"Tackles: {int(row.get('tackles', 0))}, Sacks: {row.get('sacks', 0):.1f}"

        print(f"{i+1:3}. ${val:>10,.0f} | {name:25} | {pos:4} | {school:15} | {stats}")

    print("\n" + "=" * 80)
    print("VALUATION TIER DISTRIBUTION")
    print("=" * 80)
    tier_counts = output_df['nil_tier'].value_counts()
    for tier, count in tier_counts.items():
        total_val = output_df[output_df['nil_tier'] == tier]['custom_nil_value'].sum()
        print(f"  {tier:10}: {count:6,} players | ${total_val:>15,.0f} total")

    print(f"\nTotal players: {len(output_df):,}")
    print(f"Total NIL value: ${output_df['custom_nil_value'].sum():,.0f}")
    print(f"Average value: ${output_df['custom_nil_value'].mean():,.0f}")

    return output_df


if __name__ == "__main__":
    main()
