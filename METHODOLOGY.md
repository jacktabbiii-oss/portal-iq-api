# Portal IQ Methodology

**Technical Documentation for Data Science and ML Engineering**

This document describes the methodology behind Portal IQ's machine learning models, including data sources, feature engineering decisions, model selection rationale, known limitations, and maintenance procedures.

---

## Table of Contents

1. [Data Sources](#data-sources)
2. [Feature Engineering](#feature-engineering)
3. [Model Architecture](#model-architecture)
4. [Training Procedures](#training-procedures)
5. [Known Limitations](#known-limitations)
6. [Bias Considerations](#bias-considerations)
7. [Model Maintenance](#model-maintenance)
8. [Validation Strategy](#validation-strategy)

---

## Data Sources

### Primary Data Sources

| Source | Data Type | Update Frequency | Collection Method |
|--------|-----------|------------------|-------------------|
| College Football Data API | Player stats, game logs | Weekly (in-season) | REST API |
| 247Sports | Recruiting rankings, star ratings | Daily | Web scraping |
| On3 NIL Valuations | Public NIL estimates | Weekly | Web scraping |
| Transfer Portal Database | Portal entries, commitments | Daily | Web scraping |
| Social Media APIs | Follower counts, engagement | Daily | Platform APIs |
| NFL Draft Data | Historical draft results | Annual | Public records |
| Sports Reference | Historical stats | Weekly | Web scraping |

### Data Collection Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  API Calls   │    │ Web Scraping │    │  File Import │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             ▼                               │
│                    ┌──────────────┐                         │
│                    │ Rate Limiter │                         │
│                    │   & Cache    │                         │
│                    └──────┬───────┘                         │
│                           ▼                                 │
│                    ┌──────────────┐                         │
│                    │  Validation  │                         │
│                    │   & Cleaning │                         │
│                    └──────┬───────┘                         │
│                           ▼                                 │
│                    ┌──────────────┐                         │
│                    │   data/raw/  │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Quality Measures

1. **Deduplication**: Player matching across sources using name normalization, school, and position
2. **Validation**: Range checks for all numeric fields (e.g., 40-yard time must be 4.0-6.0)
3. **Missing Data Handling**: Documented per-feature imputation strategies
4. **Freshness Tracking**: Timestamps on all records, staleness alerts

---

## Feature Engineering

### NIL Valuation Features (57 total)

#### Performance Features (18)
| Feature | Rationale | Source |
|---------|-----------|--------|
| `games_played` | Experience proxy, availability signal | CFB Stats |
| `games_started` | Starter vs. backup distinction | CFB Stats |
| `pff_grade` | Industry-standard performance metric | PFF |
| `passing_yards` | QB production | CFB Stats |
| `passing_tds` | QB scoring impact | CFB Stats |
| `completion_pct` | QB efficiency | CFB Stats |
| `qbr` | Advanced QB metric | ESPN |
| `rushing_yards` | RB/QB production | CFB Stats |
| `rushing_tds` | RB scoring impact | CFB Stats |
| `ypc` | Rushing efficiency | Derived |
| `receiving_yards` | WR/TE production | CFB Stats |
| `receiving_tds` | WR/TE scoring | CFB Stats |
| `receptions` | Target volume | CFB Stats |
| `tackles` | Defensive production | CFB Stats |
| `sacks` | Pass rush impact | CFB Stats |
| `interceptions` | Playmaking ability | CFB Stats |
| `passes_defended` | Coverage ability | CFB Stats |
| `forced_fumbles` | Playmaking ability | CFB Stats |

#### Social Media Features (8)
| Feature | Rationale | Source |
|---------|-----------|--------|
| `instagram_followers` | Primary NIL platform | Instagram API |
| `twitter_followers` | Sports engagement platform | Twitter API |
| `tiktok_followers` | Youth demographic reach | TikTok API |
| `total_followers` | Aggregate reach | Derived |
| `engagement_rate` | Content quality signal | Derived |
| `follower_growth_30d` | Momentum indicator | Derived |
| `verified_accounts` | Platform trust signal | Platform APIs |
| `content_frequency` | Activity level | Derived |

#### Recruiting Features (6)
| Feature | Rationale | Source |
|---------|-----------|--------|
| `stars` | Talent proxy | 247Sports |
| `composite_rating` | Consensus ranking | 247Sports |
| `national_rank` | Elite status indicator | 247Sports |
| `position_rank` | Position group standing | 247Sports |
| `state_rank` | Regional prominence | 247Sports |
| `early_enrollee` | Development timeline | 247Sports |

#### School/Market Features (12)
| Feature | Rationale | Source |
|---------|-----------|--------|
| `school_brand_tier` | Market size proxy (1-5) | Manual classification |
| `conference` | Media exposure level | Static mapping |
| `media_market_size` | Local NIL opportunity | Census data |
| `nil_collective_strength` | School NIL infrastructure | On3/Manual |
| `avg_attendance` | Fan engagement | CFB Stats |
| `social_media_reach` | School platform | Social APIs |
| `recent_nfl_draft_picks` | NFL pipeline prestige | Historical |
| `recent_cfp_appearances` | On-field success | Historical |
| `coaching_stability` | Program stability | Manual |
| `academic_ranking` | Non-athletic appeal | US News |
| `state_nil_laws` | Legal environment | Manual |
| `proximity_to_pro_teams` | Local market | Geographic |

#### Derived/Interaction Features (13)
| Feature | Rationale | Source |
|---------|-----------|--------|
| `performance_per_game` | Normalized production | Derived |
| `social_reach_per_start` | Efficiency metric | Derived |
| `star_rating_x_school_brand` | Talent-market interaction | Derived |
| `position_market_value` | Position premium | Historical NIL |
| `class_year_factor` | Eligibility remaining | Derived |
| `trending_score` | Recent momentum | Derived |
| `marketability_index` | Composite appeal | Derived |
| `on_field_value` | Wins above replacement proxy | Derived |
| `draft_potential` | Future earnings signal | Draft model |
| `nil_tier_encoded` | Target encoding | Derived |
| `conference_x_position` | Market-role interaction | Derived |
| `social_x_performance` | Dual-threat indicator | Derived |
| `school_x_class_year` | Development stage | Derived |

### Portal Prediction Features (31)

#### Playing Time Features (8)
| Feature | Rationale |
|---------|-----------|
| `snap_count_pct` | Usage rate - low usage = flight risk |
| `starter_flag` | Binary starter status |
| `depth_chart_position` | Positional competition |
| `snaps_trend_3game` | Recent usage trajectory |
| `expected_snaps_by_rating` | Gap between talent and usage |
| `position_room_depth` | Competition level |
| `incoming_recruits_at_position` | Future competition |
| `transfers_in_at_position` | Immediate competition |

#### Satisfaction Proxy Features (10)
| Feature | Rationale |
|---------|-----------|
| `nil_value_vs_peers` | Compensation gap |
| `nil_percentile_at_school` | Relative standing |
| `rating_vs_starters` | Talent gap |
| `hometown_distance` | Geographic ties |
| `years_at_school` | Investment/sunk cost |
| `redshirt_status` | Development path |
| `injury_history` | Career concerns |
| `position_coach_tenure` | Relationship stability |
| `academic_standing` | Transfer eligibility |
| `grad_transfer_eligible` | Transfer flexibility |

#### Program Context Features (8)
| Feature | Rationale |
|---------|-----------|
| `recent_coaching_change` | Major instability signal |
| `coordinator_change` | Scheme change risk |
| `team_win_trend` | Program trajectory |
| `sanctions_or_ncaa_issues` | Program uncertainty |
| `nil_collective_activity` | School NIL commitment |
| `recent_portal_activity` | Peer movement signal |
| `scheme_fit_score` | System compatibility |
| `culture_fit_indicators` | Qualitative factors |

#### Market Features (5)
| Feature | Rationale |
|---------|-----------|
| `player_market_value` | External demand |
| `num_schools_with_need` | Opportunity landscape |
| `position_scarcity` | Supply/demand |
| `transfer_window_timing` | Seasonal effects |
| `historic_portal_rate_by_position` | Baseline risk |

### Draft Projection Features (63)

#### Physical Measurables (15)
| Feature | Rationale |
|---------|-----------|
| `height`, `weight`, `arm_length`, `hand_size` | Position prototypes |
| `forty_yard`, `twenty_yard_split`, `ten_yard_split` | Speed/acceleration |
| `vertical_jump`, `broad_jump` | Explosiveness |
| `bench_press_reps` | Strength |
| `three_cone`, `shuttle` | Agility |
| `relative_athletic_score` | Composite athleticism |
| `height_weight_speed_score` | Size-speed balance |
| `position_prototype_match` | Ideal measurables |

#### Production Features (20)
| Feature | Rationale |
|---------|-----------|
| Per-position stats | Role-appropriate production |
| Career totals | Sustained production |
| Peak season stats | Ceiling indicator |
| Production vs. competition | Quality-adjusted stats |
| Consistency metrics | Variance in performance |

#### Advanced Metrics (15)
| Feature | Rationale |
|---------|-----------|
| `pff_career_grade` | Play-by-play evaluation |
| `wins_above_replacement` | Total impact |
| `expected_points_added` | Efficiency metric |
| `success_rate` | Down-and-distance performance |
| `pressure_rate` (pass rush) | Pass rush efficiency |
| `coverage_grade` | Secondary evaluation |
| `separation_metrics` | Receiver skills |
| `contested_catch_rate` | Ball skills |

#### Context Features (13)
| Feature | Rationale |
|---------|-----------|
| `conference_strength` | Competition level |
| `games_vs_ranked` | Quality starts |
| `all_conference_honors` | Peer recognition |
| `award_votes` | National recognition |
| `senior_bowl_invite` | Pro evaluation |
| `combine_invite` | Draft process indicator |
| `age_at_draft` | Development curve |
| `years_of_college` | Experience level |
| `pro_day_performance` | Workout results |
| `medical_flags` | Durability concerns |

---

## Model Architecture

### NIL Valuator

```
┌─────────────────────────────────────────────────────────────┐
│                     NIL Valuator Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Feature     │    │   Stacked    │    │   Value      │  │
│  │  Engineering │───▶│   Ensemble   │───▶│   Regressor  │  │
│  │  (57 feat)   │    │   (Level 1)  │    │   (Level 2)  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                             │                    │          │
│                             ▼                    ▼          │
│                      ┌──────────────┐    ┌──────────────┐  │
│                      │    Tier      │    │    SHAP      │  │
│                      │  Classifier  │    │  Explainer   │  │
│                      └──────────────┘    └──────────────┘  │
│                                                              │
│  Level 1 Models:                                            │
│  - Ridge Regression (baseline, regularization)              │
│  - Random Forest (non-linear, feature importance)           │
│  - XGBoost (gradient boosting, handles missing)             │
│  - LightGBM (fast, handles categoricals)                    │
│                                                              │
│  Level 2 Model:                                             │
│  - Ridge Regression (combines Level 1 predictions)          │
│                                                              │
│  Tier Classifier:                                           │
│  - Random Forest (multi-class: mega/premium/solid/etc)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Model Selection Rationale:**
- **Ensemble approach**: Reduces variance, combines strengths of linear and tree-based models
- **Ridge (Level 2)**: Prevents overfitting to Level 1 predictions
- **SHAP integration**: Provides interpretable value breakdowns for client communication

### Portal Predictor

```
┌─────────────────────────────────────────────────────────────┐
│                   Portal Predictor Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Feature     │    │   SMOTE      │    │   XGBoost    │  │
│  │  Engineering │───▶│   Balancing  │───▶│  Classifier  │  │
│  │  (31 feat)   │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │          │
│                                                   ▼          │
│                                          ┌──────────────┐   │
│                                          │  Calibrated  │   │
│                                          │  Probability │   │
│                                          └──────────────┘   │
│                                                              │
│  Class Imbalance Handling:                                  │
│  - Only ~15% of players enter portal annually               │
│  - SMOTE oversampling of minority class                     │
│  - Class weights in XGBoost                                 │
│  - Threshold tuning for precision/recall tradeoff           │
│                                                              │
│  Probability Calibration:                                   │
│  - Platt scaling for well-calibrated probabilities          │
│  - Important for risk communication to clients              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Model Selection Rationale:**
- **XGBoost**: Handles imbalanced data well with class weights, native missing value support
- **SMOTE**: Synthetic oversampling prevents majority class bias
- **Calibration**: Critical for communicating actual risk levels to athletic departments

### Draft Projector

```
┌─────────────────────────────────────────────────────────────┐
│                   Draft Projector Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Feature     │    │   Draft      │    │   Round      │  │
│  │  Engineering │───▶│  Classifier  │───▶│  Regressor   │  │
│  │  (63 feat)   │    │  (Drafted?)  │    │  (If drafted)│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │          │
│                             ┌──────────────────────┘          │
│                             ▼                               │
│                      ┌──────────────┐                       │
│                      │   Earnings   │                       │
│                      │  Calculator  │                       │
│                      └──────────────┘                       │
│                                                              │
│  Stage 1: Draft Classification                              │
│  - Binary: Will player be drafted? (RF Classifier)          │
│  - Outputs probability of being drafted                     │
│                                                              │
│  Stage 2: Round Prediction (conditional)                    │
│  - Ordinal regression for rounds 1-7                        │
│  - Only applied if draft probability > threshold            │
│                                                              │
│  Stage 3: Financial Projection                              │
│  - Jimmy Johnson Trade Value Chart for pick value           │
│  - NFL Rookie Wage Scale for contract estimates             │
│  - Career earnings projection based on position/round       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Win Impact Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Win Impact Model                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Roster     │    │  Position    │    │    Win       │  │
│  │  Aggregation │───▶│   Weights    │───▶│  Projection  │  │
│  │              │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│  Position Impact Weights:                                   │
│  - QB: 25% of team variance                                 │
│  - OL (aggregate): 15%                                      │
│  - WR/TE (aggregate): 12%                                   │
│  - RB: 8%                                                   │
│  - DL (aggregate): 12%                                      │
│  - LB (aggregate): 10%                                      │
│  - DB (aggregate): 10%                                      │
│  - Special Teams: 5%                                        │
│  - Coaching/Scheme: 3% (external factor)                    │
│                                                              │
│  Model: Ridge Regression on aggregated roster strength      │
│  Trained on historical team ratings → actual wins           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Roster Optimizer

```
┌─────────────────────────────────────────────────────────────┐
│                    Roster Optimizer (PuLP)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Objective Function:                                        │
│  Maximize: Σ (player_value × selected) - risk_penalty       │
│                                                              │
│  Subject to:                                                │
│  - Σ (nil_cost × selected) ≤ total_budget                  │
│  - Σ (selected at position) ≥ min_position_requirements    │
│  - Σ (selected at position) ≤ max_position_limits          │
│  - scholarship_count ≤ 85                                   │
│  - For each player: selected ∈ {0, 1}                      │
│                                                              │
│  Value Function:                                            │
│  player_value = win_impact × (1 - flight_risk × risk_weight)│
│                                                              │
│  Solver: CBC (COIN-OR Branch and Cut)                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Procedures

### Data Splitting Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                  Temporal Train/Test Split                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Training Data: 2019-2023 seasons                           │
│  Validation Data: 2023 season (walk-forward)                │
│  Test Data: 2024 season (holdout)                           │
│                                                              │
│  Why Temporal Split:                                        │
│  - Prevents data leakage from future information            │
│  - Mirrors real-world deployment (predicting future)        │
│  - NIL market evolving rapidly, recency matters             │
│                                                              │
│  ──────────────────────────────────────────────────────────│
│  │  2019  │  2020  │  2021  │  2022  │  2023  │  2024  │   │
│  │        │        │        │        │        │        │   │
│  │◄────── Training Data ──────────────►│ Val  │  Test │   │
│  ──────────────────────────────────────────────────────────│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Validation

- **NIL Model**: 5-fold time-series cross-validation
- **Portal Model**: Stratified K-Fold (preserves class balance)
- **Draft Model**: Leave-one-season-out cross-validation

### Hyperparameter Tuning

| Model | Method | Key Parameters |
|-------|--------|----------------|
| Ridge | GridSearchCV | alpha: [0.01, 0.1, 1, 10, 100] |
| Random Forest | RandomizedSearchCV | n_estimators, max_depth, min_samples_split |
| XGBoost | Optuna | learning_rate, max_depth, subsample, colsample_bytree |
| LightGBM | Optuna | num_leaves, learning_rate, feature_fraction |

### Training Pipeline

```bash
# Full training pipeline
python scripts/run_pipeline.py --collect-data --train-models

# Individual model training
python -c "from models.nil_valuator import NILValuator; NILValuator().train(...)"
```

---

## Known Limitations

### Data Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Small NIL training sample** | NIL market only exists since July 2021; limited historical data | Use transfer learning from NFL contract data; regularization to prevent overfitting |
| **Self-reported NIL values** | Public NIL estimates may be inaccurate or inflated | Use multiple sources, weight by source reliability, model uncertainty |
| **Incomplete social media data** | Not all platforms accessible, private accounts | Impute from available data, use school/position averages |
| **Survivorship bias in portal data** | Only see successful transfers, not those who stayed | Weight by school/position portal rates |
| **Draft data selection bias** | Only drafted players have known outcomes | Two-stage model: first predict if drafted |

### Model Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Assumes past patterns continue** | NIL market rapidly evolving | Frequent retraining, recency weighting |
| **Position-specific models not trained** | Generic model may miss position nuances | Position-specific feature engineering |
| **Does not model agent/agency effects** | Representation quality affects NIL deals | Future enhancement |
| **Injury impact not fully captured** | Career-altering injuries hard to predict | Include injury history features |
| **Transfer fit prediction is noisy** | Scheme fit is subjective | Focus on measurable factors |

### What the Model Cannot Predict

1. **Scandals or off-field issues**: Unpredictable events that tank NIL value
2. **Coaching changes**: Future staff turnover unknown
3. **Injury occurrences**: Can model risk factors but not specific injuries
4. **Viral moments**: One viral video can dramatically change social following
5. **Conference realignment effects**: Long-term structural changes to CFB

---

## Bias Considerations

### School Brand Bias

**Problem**: Model may systematically overvalue players at "blue blood" programs (Alabama, Ohio State, etc.) because:
- Historical NIL data skews toward large-school players
- School brand is a legitimate NIL factor, but model may overweight it
- Small-school stars may be undervalued

**Mitigation Strategies**:
1. Include school brand as explicit feature (allows interpretation)
2. Train separate models for P5 vs. G5 schools
3. Use residual analysis to identify school-level bias
4. Include "performance vs. competition" features

**Monitoring**:
```python
# Check for school bias in predictions
predictions.groupby('school_brand_tier').agg({
    'predicted_value': 'mean',
    'actual_value': 'mean',
    'error': 'mean'
})
```

### Position Bias

**Problem**: QBs and skill position players dominate NIL market; model may:
- Undervalue elite linemen
- Not capture position scarcity dynamics
- Miss specialist value (kickers, long snappers)

**Mitigation Strategies**:
1. Position-specific value floors
2. Position interaction features
3. Include position scarcity in market features
4. Separate tier thresholds by position group

### Recruiting Service Bias

**Problem**: 247Sports/Rivals ratings have known biases:
- Geographic bias (California, Texas, Florida overrated)
- School visit bias (more visits = higher rating)
- Late bloomers underrated

**Mitigation Strategies**:
1. Use composite ratings (average multiple services)
2. Weight college production more heavily than recruiting rating for upperclassmen
3. Include "exceeded/met/missed expectations" as a feature

### Demographic Considerations

**Principle**: Model uses only performance and market factors, not demographic features.

**Audited for**:
- No direct demographic features (race, ethnicity, religion)
- Proxy variables checked (hometown, high school)
- Geographic features used only for market size, not demographic proxies

---

## Model Maintenance

### Retraining Schedule

| Model | Frequency | Trigger Events |
|-------|-----------|----------------|
| NIL Valuator | Monthly | Season end, major NIL deals announced |
| Portal Predictor | Monthly | Portal window opens/closes |
| Draft Projector | Annual | After NFL Draft |
| Win Impact | Annual | After season ends |

### Performance Monitoring

```python
# Weekly monitoring script
from sklearn.metrics import mean_absolute_percentage_error

# Load recent predictions and actuals
predictions = load_recent_predictions()
actuals = load_actual_nil_values()

# Calculate metrics
mape = mean_absolute_percentage_error(actuals, predictions)
tier_accuracy = (predictions['tier'] == actuals['tier']).mean()

# Alert if degradation
if mape > 0.30:  # 30% MAPE threshold
    send_alert("NIL model MAPE exceeded threshold")

if tier_accuracy < 0.65:  # 65% accuracy threshold
    send_alert("NIL tier accuracy below threshold")
```

### Model Versioning

```
models/
├── nil_valuator/
│   ├── v1.0.0_2024-01-15/
│   │   ├── model.pkl
│   │   ├── metadata.json
│   │   └── performance.json
│   ├── v1.1.0_2024-03-01/
│   └── current -> v1.1.0_2024-03-01/
```

### Data Refresh Procedure

1. **Daily**: Social media metrics, portal entries
2. **Weekly**: Player statistics (in-season)
3. **Monthly**: NIL valuations, recruiting updates
4. **Quarterly**: Full model retraining
5. **Annually**: Feature engineering review, new data source evaluation

### Adding New Features

1. Document rationale in this methodology doc
2. Add to appropriate feature engineering module
3. Run correlation analysis against existing features
4. Test on validation set before production
5. A/B test in shadow mode before full deployment

---

## Validation Strategy

### Offline Validation Metrics

| Model | Primary Metric | Secondary Metrics |
|-------|---------------|-------------------|
| NIL Valuator | MAPE | R², Tier Accuracy, Calibration |
| Portal Predictor | AUC-ROC | Precision@K, Recall@50%, Brier Score |
| Draft Projector | MAE (round) | Draft Accuracy, Round ±1 Accuracy |
| Win Impact | RMSE | R², Within-2-Wins Accuracy |

### Online Validation

- **A/B Testing**: New models run in shadow mode before promotion
- **Prediction Logs**: All predictions logged with timestamps for retrospective analysis
- **Client Feedback Loop**: Track which predictions clients question or override

### Backtesting

```python
# Walk-forward backtesting
for test_year in [2022, 2023, 2024]:
    train_data = data[data['season'] < test_year]
    test_data = data[data['season'] == test_year]

    model = NILValuator()
    model.train(train_data)
    predictions = model.predict(test_data)

    metrics[test_year] = calculate_metrics(predictions, test_data)
```

---

## Appendix: Feature Importance Rankings

### Top 10 NIL Features (by SHAP importance)

1. `total_followers` - Social media reach
2. `school_brand_tier` - Program prestige
3. `position_market_value` - Position premium
4. `stars` - Recruiting pedigree
5. `games_started` - On-field role
6. `pff_grade` - Performance quality
7. `conference` - Media exposure
8. `engagement_rate` - Social media quality
9. `class_year_factor` - Eligibility remaining
10. `nil_collective_strength` - School NIL infrastructure

### Top 10 Portal Risk Features

1. `snap_count_pct` - Playing time
2. `starter_flag` - Role clarity
3. `nil_value_vs_peers` - Compensation gap
4. `recent_coaching_change` - Stability
5. `rating_vs_starters` - Talent/usage mismatch
6. `incoming_recruits_at_position` - Future competition
7. `years_at_school` - Investment level
8. `hometown_distance` - Geographic ties
9. `nil_percentile_at_school` - Relative standing
10. `team_win_trend` - Program trajectory

---

*Last Updated: February 2026*
*Version: 1.0*
*Author: Portal IQ Data Science Team*
