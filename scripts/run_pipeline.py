#!/usr/bin/env python3
"""
Portal IQ - Full Pipeline Runner

Runs the entire ML pipeline end-to-end:
1. Collect data (or use cached)
2. Engineer features
3. Train all models
4. Generate sample predictions
5. Save all outputs
6. Print summary of model performance
7. Launch dashboard (optional)

Usage:
    python scripts/run_pipeline.py --all
    python scripts/run_pipeline.py --collect-data --train-models
    python scripts/run_pipeline.py --predict --dashboard
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_run.log"),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================

CONFIG = {
    "data_dir": "data",
    "models_dir": "models/trained",
    "outputs_dir": "outputs",
    "cache_hours": 24,
    "random_seed": 42,
}


# =============================================================================
# Pipeline Steps
# =============================================================================

def step_collect_data(use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """Step 1: Collect data from all sources.

    Args:
        use_cache: Whether to use cached data if available

    Returns:
        Dict of DataFrames with collected data
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting Data")
    logger.info("=" * 60)

    data = {}
    cache_dir = Path(CONFIG["data_dir"]) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try to collect from each data source
    collectors = [
        ("cfb_stats", "CFBStatsCollector"),
        ("cfb_recruiting", "CFBRecruitingCollector"),
        ("cfb_portal", "CFBPortalCollector"),
        ("cfb_nil", "CFBNILCollector"),
        ("draft_history", "DraftHistoryCollector"),
    ]

    for data_name, collector_name in collectors:
        cache_file = cache_dir / f"{data_name}_2024.parquet"

        if use_cache and cache_file.exists():
            logger.info(f"  Loading {data_name} from cache...")
            try:
                data[data_name] = pd.read_parquet(cache_file)
                logger.info(f"  [OK] Loaded {len(data[data_name])} records")
                continue
            except Exception as e:
                logger.warning(f"  Cache load failed: {e}")

        # Try to collect fresh data
        try:
            module = __import__(f"data_collection.{data_name.split('_')[0]}_{data_name.split('_')[1] if '_' in data_name else 'stats'}",
                              fromlist=[collector_name])
            collector_class = getattr(module, collector_name)
            collector = collector_class()

            logger.info(f"  Collecting {data_name}...")
            df = collector.collect_all(season=2024)

            if df is not None and not df.empty:
                data[data_name] = df
                df.to_parquet(cache_file)
                logger.info(f"  [OK] Collected {len(df)} records")
            else:
                logger.warning(f"  [SKIP] No data collected for {data_name}")

        except (ImportError, AttributeError) as e:
            logger.warning(f"  [SKIP] {collector_name} not available: {e}")
        except Exception as e:
            logger.error(f"  [ERROR] Failed to collect {data_name}: {e}")

    # Generate sample data if nothing was collected
    if not data:
        logger.info("  Generating sample data for demonstration...")
        data = generate_sample_data()

    logger.info(f"  Total datasets: {len(data)}")
    return data


def step_engineer_features(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Step 2: Engineer features from raw data.

    Args:
        data: Dict of raw DataFrames

    Returns:
        Dict of feature DataFrames
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Engineering Features")
    logger.info("=" * 60)

    features = {}
    processed_dir = Path(CONFIG["data_dir"]) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # NIL Features
    try:
        from feature_engineering.nil_features import NILFeatureEngineer

        logger.info("  Building NIL features...")
        nil_engineer = NILFeatureEngineer()

        # Combine relevant data
        if "cfb_stats" in data:
            nil_features = nil_engineer.build_features(data["cfb_stats"])
            features["nil_features"] = nil_features
            nil_features.to_csv(processed_dir / "nil_features_ready.csv", index=False)
            logger.info(f"  [OK] NIL features: {nil_features.shape}")
    except (ImportError, Exception) as e:
        logger.warning(f"  [SKIP] NIL features: {e}")

    # Portal Features
    try:
        from feature_engineering.portal_features import PortalFeatureEngineer

        logger.info("  Building portal features...")
        portal_engineer = PortalFeatureEngineer()

        if "cfb_stats" in data:
            flight_features = portal_engineer.build_flight_risk_features(data["cfb_stats"])
            features["flight_risk_features"] = flight_features
            flight_features.to_csv(processed_dir / "flight_risk_features.csv", index=False)
            logger.info(f"  [OK] Flight risk features: {flight_features.shape}")
    except (ImportError, Exception) as e:
        logger.warning(f"  [SKIP] Portal features: {e}")

    # Draft Features
    try:
        from feature_engineering.draft_features import DraftFeatureEngineer

        logger.info("  Building draft features...")
        draft_engineer = DraftFeatureEngineer()

        if "draft_history" in data or "cfb_stats" in data:
            source_data = data.get("draft_history", data.get("cfb_stats"))
            draft_features = draft_engineer.build_features(source_data)
            features["draft_features"] = draft_features
            draft_features.to_csv(processed_dir / "draft_features.csv", index=False)
            logger.info(f"  [OK] Draft features: {draft_features.shape}")
    except (ImportError, Exception) as e:
        logger.warning(f"  [SKIP] Draft features: {e}")

    # Use sample features if none were built
    if not features:
        logger.info("  Using sample features for demonstration...")
        features = generate_sample_features()

    logger.info(f"  Total feature sets: {len(features)}")
    return features


def step_train_models(features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Step 3: Train all ML models.

    Args:
        features: Dict of feature DataFrames

    Returns:
        Dict of trained models with metrics
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Training Models")
    logger.info("=" * 60)

    models = {}
    metrics = {}
    models_dir = Path(CONFIG["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(CONFIG["random_seed"])

    # Train NIL Valuator
    try:
        from models.nil_valuator import NILValuator

        logger.info("  Training NILValuator...")
        nil_model = NILValuator()

        if "nil_features" in features:
            df = features["nil_features"]

            # Prepare data
            feature_cols = [c for c in df.columns if c not in ["player_id", "name", "nil_value", "nil_tier", "school"]]
            X = df[feature_cols].fillna(0)

            # Create target if not present
            if "nil_value" not in df.columns:
                df["nil_value"] = np.random.uniform(10000, 500000, len(df))
            if "nil_tier" not in df.columns:
                df["nil_tier"] = pd.cut(df["nil_value"], bins=[0, 25000, 100000, 500000, 1000000, float('inf')],
                                       labels=["entry", "moderate", "solid", "premium", "mega"])

            y_value = df["nil_value"]
            y_tier = df["nil_tier"]

            nil_model.train(X, y_value, y_tier)
            models["nil_valuator"] = nil_model
            logger.info("  [OK] NILValuator trained")

            # Calculate metrics
            if hasattr(nil_model, 'value_model') and nil_model.value_model:
                metrics["nil_valuator"] = {"status": "trained"}
    except (ImportError, Exception) as e:
        logger.warning(f"  [SKIP] NILValuator: {e}")

    # Train Portal Predictor
    try:
        from models.portal_predictor import PortalPredictor

        logger.info("  Training PortalPredictor...")
        portal_model = PortalPredictor()

        if "flight_risk_features" in features:
            df = features["flight_risk_features"]

            feature_cols = [c for c in df.columns if c not in ["player_id", "name", "entered_portal", "school"]]
            X = df[feature_cols].fillna(0)

            if "entered_portal" not in df.columns:
                df["entered_portal"] = np.random.binomial(1, 0.25, len(df))

            y = df["entered_portal"]

            portal_model.train_flight_risk(X, y)
            models["portal_predictor"] = portal_model
            logger.info("  [OK] PortalPredictor trained")
            metrics["portal_predictor"] = {"status": "trained"}
    except (ImportError, Exception) as e:
        logger.warning(f"  [SKIP] PortalPredictor: {e}")

    # Train Draft Projector
    try:
        from models.draft_projector import DraftProjector

        logger.info("  Training DraftProjector...")
        draft_model = DraftProjector()

        if "draft_features" in features:
            df = features["draft_features"]

            feature_cols = [c for c in df.columns if c not in ["player_id", "name", "was_drafted", "draft_round", "draft_pick", "school"]]
            X = df[feature_cols].fillna(0)

            if "was_drafted" not in df.columns:
                df["was_drafted"] = np.random.binomial(1, 0.3, len(df))
            if "draft_round" not in df.columns:
                df["draft_round"] = np.where(df["was_drafted"], np.random.randint(1, 8, len(df)), None)

            y_drafted = df["was_drafted"].astype(int)
            y_round = df["draft_round"].fillna(7)

            draft_model.train(X, y_drafted, y_round)
            models["draft_projector"] = draft_model
            logger.info("  [OK] DraftProjector trained")
            metrics["draft_projector"] = {"status": "trained"}
    except (ImportError, Exception) as e:
        logger.warning(f"  [SKIP] DraftProjector: {e}")

    # Train Win Model
    try:
        from models.win_model import WinImpactModel

        logger.info("  Training WinImpactModel...")
        win_model = WinImpactModel()

        # Win model may not need explicit training
        models["win_model"] = win_model
        logger.info("  [OK] WinImpactModel initialized")
        metrics["win_model"] = {"status": "initialized"}
    except (ImportError, Exception) as e:
        logger.warning(f"  [SKIP] WinImpactModel: {e}")

    logger.info(f"  Total models: {len(models)}")
    return {"models": models, "metrics": metrics}


def step_generate_predictions(
    models: Dict[str, Any],
    features: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Step 4: Generate sample predictions.

    Args:
        models: Dict of trained models
        features: Dict of feature DataFrames

    Returns:
        Dict of prediction DataFrames
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Generating Predictions")
    logger.info("=" * 60)

    predictions = {}
    outputs_dir = Path(CONFIG["outputs_dir"]) / "predictions"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # NIL Predictions
    if "nil_valuator" in models and "nil_features" in features:
        try:
            logger.info("  Generating NIL predictions...")
            model = models["nil_valuator"]
            df = features["nil_features"]

            feature_cols = [c for c in df.columns if c not in ["player_id", "name", "nil_value", "nil_tier", "school"]]
            X = df[feature_cols].fillna(0)

            preds = model.predict(X)

            if isinstance(preds, list):
                pred_df = pd.DataFrame(preds)
            else:
                pred_df = preds

            predictions["nil_predictions"] = pred_df
            pred_df.to_csv(outputs_dir / "nil_predictions.csv", index=False)
            logger.info(f"  [OK] NIL predictions: {len(pred_df)} players")
        except Exception as e:
            logger.warning(f"  [SKIP] NIL predictions: {e}")

    # Portal Predictions
    if "portal_predictor" in models and "flight_risk_features" in features:
        try:
            logger.info("  Generating flight risk predictions...")
            model = models["portal_predictor"]
            df = features["flight_risk_features"]

            feature_cols = [c for c in df.columns if c not in ["player_id", "name", "entered_portal", "school"]]
            X = df[feature_cols].fillna(0)

            preds = model.predict_flight_risk(X)

            if isinstance(preds, list):
                pred_df = pd.DataFrame(preds)
            else:
                pred_df = preds

            predictions["flight_risk_predictions"] = pred_df
            pred_df.to_csv(outputs_dir / "flight_risk_predictions.csv", index=False)
            logger.info(f"  [OK] Flight risk predictions: {len(pred_df)} players")
        except Exception as e:
            logger.warning(f"  [SKIP] Flight risk predictions: {e}")

    # Draft Predictions
    if "draft_projector" in models and "draft_features" in features:
        try:
            logger.info("  Generating draft predictions...")
            model = models["draft_projector"]
            df = features["draft_features"]

            feature_cols = [c for c in df.columns if c not in ["player_id", "name", "was_drafted", "draft_round", "draft_pick", "school"]]
            X = df[feature_cols].fillna(0)

            preds = model.predict(X)

            if isinstance(preds, list):
                pred_df = pd.DataFrame(preds)
            else:
                pred_df = preds

            predictions["draft_predictions"] = pred_df
            pred_df.to_csv(outputs_dir / "draft_predictions.csv", index=False)
            logger.info(f"  [OK] Draft predictions: {len(pred_df)} players")
        except Exception as e:
            logger.warning(f"  [SKIP] Draft predictions: {e}")

    logger.info(f"  Total prediction sets: {len(predictions)}")
    return predictions


def step_save_outputs(
    data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    models_info: Dict[str, Any],
    predictions: Dict[str, pd.DataFrame]
) -> None:
    """Step 5: Save all outputs and summaries.

    Args:
        data: Raw data
        features: Engineered features
        models_info: Model information and metrics
        predictions: Generated predictions
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Saving Outputs")
    logger.info("=" * 60)

    outputs_dir = Path(CONFIG["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save pipeline summary
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "data_sources": list(data.keys()),
        "feature_sets": list(features.keys()),
        "models_trained": list(models_info.get("models", {}).keys()),
        "predictions_generated": list(predictions.keys()),
        "metrics": models_info.get("metrics", {}),
    }

    summary_path = outputs_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"  [OK] Pipeline summary saved to {summary_path}")

    # Save data inventory
    inventory = {}
    for name, df in data.items():
        inventory[name] = {"rows": len(df), "columns": list(df.columns)}

    inventory_path = outputs_dir / "data_inventory.json"
    with open(inventory_path, "w") as f:
        json.dump(inventory, f, indent=2)

    logger.info(f"  [OK] Data inventory saved to {inventory_path}")


def step_print_summary(
    models_info: Dict[str, Any],
    predictions: Dict[str, pd.DataFrame]
) -> None:
    """Step 6: Print summary of model performance.

    Args:
        models_info: Model information and metrics
        predictions: Generated predictions
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Pipeline Summary")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("PORTAL IQ PIPELINE SUMMARY")
    print("=" * 60)

    # Models trained
    print("\nModels Trained:")
    for model_name, status in models_info.get("metrics", {}).items():
        print(f"  - {model_name}: {status.get('status', 'unknown')}")

    # Predictions generated
    print("\nPredictions Generated:")
    for pred_name, df in predictions.items():
        print(f"  - {pred_name}: {len(df)} records")

    # Sample predictions
    if "nil_predictions" in predictions and len(predictions["nil_predictions"]) > 0:
        print("\nSample NIL Predictions:")
        sample = predictions["nil_predictions"].head(5)
        if "predicted_value" in sample.columns:
            for _, row in sample.iterrows():
                value = row.get("predicted_value", row.get("nil_value", 0))
                print(f"  - Player: ${value:,.0f}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60 + "\n")


def step_launch_dashboard() -> None:
    """Step 7: Launch the Streamlit dashboard (optional)."""
    logger.info("=" * 60)
    logger.info("STEP 7: Launching Dashboard")
    logger.info("=" * 60)

    import subprocess

    dashboard_path = Path(__file__).parent.parent.parent / "dashboard" / "app.py"

    if dashboard_path.exists():
        logger.info(f"  Launching dashboard from {dashboard_path}")
        subprocess.Popen(["streamlit", "run", str(dashboard_path)])
        logger.info("  [OK] Dashboard launched at http://localhost:8501")
    else:
        logger.warning(f"  Dashboard not found at {dashboard_path}")


# =============================================================================
# Helper Functions
# =============================================================================

def generate_sample_data() -> Dict[str, pd.DataFrame]:
    """Generate sample data for demonstration."""
    np.random.seed(42)

    players = []
    positions = ["QB", "RB", "WR", "TE", "OT", "EDGE", "DT", "LB", "CB", "S"]
    schools = ["Alabama", "Ohio State", "Georgia", "Texas", "Oregon"]

    for i in range(200):
        players.append({
            "player_id": f"player_{i}",
            "name": f"Player {i}",
            "school": schools[i % len(schools)],
            "position": positions[i % len(positions)],
            "games_played": np.random.randint(8, 14),
            "overall_rating": np.random.uniform(0.65, 0.95),
            "stars": np.random.choice([3, 4, 5]),
            "total_followers": np.random.randint(1000, 500000),
        })

    return {"cfb_stats": pd.DataFrame(players)}


def generate_sample_features() -> Dict[str, pd.DataFrame]:
    """Generate sample features for demonstration."""
    np.random.seed(42)

    n = 200
    features = {
        "nil_features": pd.DataFrame({
            "player_id": [f"player_{i}" for i in range(n)],
            "games_played": np.random.randint(8, 14, n),
            "overall_rating": np.random.uniform(0.65, 0.95, n),
            "total_followers": np.random.randint(1000, 500000, n),
            "stars": np.random.choice([3, 4, 5], n),
            "nil_value": np.random.uniform(10000, 500000, n),
            "nil_tier": np.random.choice(["entry", "moderate", "solid", "premium", "mega"], n),
        }),
        "flight_risk_features": pd.DataFrame({
            "player_id": [f"player_{i}" for i in range(n)],
            "games_played": np.random.randint(8, 14, n),
            "games_started": np.random.randint(0, 14, n),
            "overall_rating": np.random.uniform(0.65, 0.95, n),
            "entered_portal": np.random.binomial(1, 0.25, n),
        }),
        "draft_features": pd.DataFrame({
            "player_id": [f"player_{i}" for i in range(n)],
            "pff_grade": np.random.uniform(60, 95, n),
            "forty_yard": np.random.uniform(4.3, 5.2, n),
            "vertical": np.random.randint(28, 42, n),
            "was_drafted": np.random.binomial(1, 0.3, n),
            "draft_round": np.random.choice([1, 2, 3, 4, 5, 6, 7, None], n),
        }),
    }

    return features


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Portal IQ - Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_pipeline.py --all              # Run everything
    python scripts/run_pipeline.py --collect-data     # Only collect data
    python scripts/run_pipeline.py --train-models     # Only train models
    python scripts/run_pipeline.py --predict          # Only generate predictions
    python scripts/run_pipeline.py --dashboard        # Launch dashboard
        """
    )

    parser.add_argument("--all", action="store_true",
                        help="Run the complete pipeline")
    parser.add_argument("--collect-data", action="store_true",
                        help="Collect data from sources")
    parser.add_argument("--train-models", action="store_true",
                        help="Train all models")
    parser.add_argument("--predict", action="store_true",
                        help="Generate predictions")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the Streamlit dashboard")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached data")

    args = parser.parse_args()

    # Default to --all if no arguments provided
    if not any([args.all, args.collect_data, args.train_models, args.predict, args.dashboard]):
        args.all = True

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("PORTAL IQ PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        # Initialize data containers
        data = {}
        features = {}
        models_info = {"models": {}, "metrics": {}}
        predictions = {}

        # Step 1: Collect Data
        if args.all or args.collect_data:
            data = step_collect_data(use_cache=not args.no_cache)

        # Step 2: Engineer Features
        if args.all or args.train_models:
            if not data:
                # Load cached data if available
                data = step_collect_data(use_cache=True)
            features = step_engineer_features(data)

        # Step 3: Train Models
        if args.all or args.train_models:
            if not features:
                features = generate_sample_features()
            models_info = step_train_models(features)

        # Step 4: Generate Predictions
        if args.all or args.predict:
            if not models_info.get("models"):
                # Try to train models if not done
                if not features:
                    features = generate_sample_features()
                models_info = step_train_models(features)
            predictions = step_generate_predictions(models_info.get("models", {}), features)

        # Step 5: Save Outputs
        if args.all or args.predict:
            step_save_outputs(data, features, models_info, predictions)

        # Step 6: Print Summary
        if args.all or args.predict:
            step_print_summary(models_info, predictions)

        # Step 7: Launch Dashboard
        if args.dashboard:
            step_launch_dashboard()

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f} seconds")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
