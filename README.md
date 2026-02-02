# Portal IQ - AI-Powered College Football Intelligence

**The most advanced NIL valuation, transfer portal analytics, and roster optimization platform for college football.**

Portal IQ gives athletic departments, NIL collectives, agents, and sports analytics companies the intelligence they need to make data-driven decisions in the modern era of college athletics.

---

## Value Proposition

| Problem | Portal IQ Solution |
|---------|-------------------|
| NIL valuations are subjective and inconsistent | ML-powered valuations using 60+ features |
| Teams blindly lose players to the transfer portal | Flight risk prediction with retention recommendations |
| Finding portal fits is manual and time-consuming | AI-scored portal fit matching |
| NIL budgets are allocated inefficiently | Linear programming budget optimization |
| Draft potential isn't factored into NIL decisions | Draft projection integrated with NIL valuation |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PORTAL IQ PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Dashboard     │    │    FastAPI      │    │   PlaymakerVC   │         │
│  │   (Streamlit)   │───▶│    REST API     │◀───│   Integration   │         │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘         │
│                                  │                                          │
│  ┌───────────────────────────────┴───────────────────────────────┐         │
│  │                        ML MODELS                               │         │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │         │
│  │  │NILValuator  │ │PortalPred.  │ │DraftProject.│ │WinImpact │ │         │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘ │         │
│  │  ┌─────────────────────────────────────────────────────────┐  │         │
│  │  │              RosterOptimizer (PuLP LP)                   │  │         │
│  │  └─────────────────────────────────────────────────────────┘  │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                  │                                          │
│  ┌───────────────────────────────┴───────────────────────────────┐         │
│  │                   FEATURE ENGINEERING                          │         │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │         │
│  │  │NILFeatures  │ │PortalFeat.  │ │DraftFeatures│              │         │
│  │  │  (57 feat)  │ │  (31 feat)  │ │  (63 feat)  │              │         │
│  │  └─────────────┘ └─────────────┘ └─────────────┘              │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                  │                                          │
│  ┌───────────────────────────────┴───────────────────────────────┐         │
│  │                    DATA COLLECTION                             │         │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │         │
│  │  │CFB Stats │ │Recruiting│ │  Portal  │ │   NIL    │         │         │
│  │  │Collector │ │Collector │ │Collector │ │Collector │         │         │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Screenshots

> **TODO**: Add screenshots of:
> - Dashboard home page
> - NIL Valuator with prediction breakdown
> - Portal Intelligence flight risk report
> - Draft Tracker projections
> - Roster Builder optimization results

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/elitesportssolutions/portal-iq.git
cd portal-iq/ml-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 5. Collect data (or use cached sample data)
python scripts/run_pipeline.py --collect-data

# 6. Train models
python scripts/run_pipeline.py --train-models

# 7. Launch the dashboard
python scripts/run_pipeline.py --dashboard

# Or run everything at once:
python scripts/run_pipeline.py --all
```

### Running the API

```bash
# Start the API server
cd src
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# API will be available at:
# - http://localhost:8000 (health check)
# - http://localhost:8000/docs (Swagger UI)
# - http://localhost:8000/redoc (ReDoc)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific test files
pytest tests/test_models/test_nil_valuator.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## API Documentation

### Authentication

All API endpoints (except health check) require an API key:

```bash
curl -X POST "http://localhost:8000/api/nil/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"player": {"name": "Test", "school": "Alabama", "position": "QB"}}'
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check (no auth) |
| `/api/nil/predict` | POST | Get NIL valuation |
| `/api/nil/transfer-impact` | POST | Analyze transfer value change |
| `/api/nil/market-report` | POST | Get market overview |
| `/api/portal/flight-risk` | POST | Predict flight risk |
| `/api/portal/team-report` | POST | Get team-wide risk report |
| `/api/portal/fit-score` | POST | Calculate portal fit |
| `/api/portal/recommendations` | POST | Get portal targets |
| `/api/draft/project` | POST | Get draft projection |
| `/api/draft/mock` | POST | Generate mock draft |
| `/api/roster/optimize` | POST | Optimize NIL budget |
| `/api/roster/scenario` | POST | Analyze roster changes |
| `/api/roster/{school}/report` | GET | Get full roster report |

### Example Requests

#### NIL Prediction

```bash
curl -X POST "http://localhost:8000/api/nil/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "player": {
      "name": "Arch Manning",
      "school": "Texas",
      "position": "QB",
      "class_year": "Sophomore",
      "stats": {
        "games_played": 12,
        "passing_yards": 3200,
        "passing_tds": 28
      },
      "social_media": {
        "instagram_followers": 500000,
        "twitter_followers": 200000
      },
      "recruiting": {
        "stars": 5,
        "national_rank": 1
      },
      "overall_rating": 0.95
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "player_name": "Arch Manning",
    "predicted_value": 3200000,
    "value_tier": "mega",
    "tier_probabilities": {"mega": 0.85, "premium": 0.12, "solid": 0.03},
    "confidence": 0.88,
    "value_breakdown": {
      "base_value": 1200000,
      "social_media_premium": 800000,
      "school_brand_factor": 600000,
      "position_market_factor": 400000,
      "draft_potential_premium": 200000
    }
  }
}
```

#### Flight Risk

```bash
curl -X POST "http://localhost:8000/api/portal/flight-risk" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "player": {
      "name": "Backup QB",
      "school": "Florida State",
      "position": "QB",
      "class_year": "Junior",
      "overall_rating": 0.82
    },
    "team_context": {
      "recent_coaching_change": true
    }
  }'
```

#### Roster Optimization

```bash
curl -X POST "http://localhost:8000/api/roster/optimize" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "school": "Georgia",
    "total_budget": 15000000,
    "win_target": 11
  }'
```

---

## Model Performance

> **Note**: These metrics are placeholders. Update after training on production data.

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| NILValuator | Value Regression | MAPE | ~25% |
| NILValuator | Tier Classification | Accuracy | ~72% |
| PortalPredictor | Flight Risk | AUC-ROC | ~0.78 |
| PortalPredictor | Flight Risk | Recall@50% | ~65% |
| DraftProjector | Draft Classification | Accuracy | ~68% |
| DraftProjector | Round Prediction | MAE | ~1.2 rounds |
| WinImpactModel | Win Projection | RMSE | ~1.5 wins |

---

## PlaymakerVC Integration Guide

Portal IQ is designed for seamless integration with PlaymakerVC's client management platform.

### Quick Integration

1. **Get API Key**: Contact Elite Sports Solutions for your API key
2. **Base URL**: `https://api.portaliq.app` (production)
3. **Authentication**: Include `X-API-Key` header in all requests

### Recommended Integration Points

| PlaymakerVC Feature | Portal IQ Endpoint | Use Case |
|--------------------|--------------------|----------|
| Client Profile | `/api/nil/predict` | Show NIL valuation on profile page |
| Client List | `/api/nil/predict` (batch) | Bulk valuations for all clients |
| Transfer Advisor | `/api/nil/transfer-impact` | Show value change for potential transfers |
| School Dashboard | `/api/portal/team-report` | Show flight risk for team's roster |
| Recruiting Targets | `/api/portal/recommendations` | Suggest portal targets |

### Code Example (JavaScript)

```javascript
const PORTAL_IQ_API = 'https://api.portaliq.app';
const API_KEY = process.env.PORTAL_IQ_API_KEY;

async function getNILValuation(player) {
  const response = await fetch(`${PORTAL_IQ_API}/api/nil/predict`, {
    method: 'POST',
    headers: {
      'X-API-Key': API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ player }),
  });

  const data = await response.json();
  return data.data;
}

// Usage
const valuation = await getNILValuation({
  name: 'Player Name',
  school: 'Alabama',
  position: 'QB',
  // ... other fields
});

console.log(`NIL Value: $${valuation.predicted_value.toLocaleString()}`);
```

---

## Project Structure

```
ml-engine/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── app.py              # Main app, middleware, health checks
│   │   ├── routes.py           # All API endpoints
│   │   └── schemas.py          # Pydantic request/response models
│   │
│   ├── data_collection/        # Data collectors
│   │   ├── cfb_stats.py        # Player statistics
│   │   ├── cfb_recruiting.py   # Recruiting data
│   │   ├── cfb_portal.py       # Portal entries
│   │   └── cfb_nil.py          # NIL valuations
│   │
│   ├── feature_engineering/    # Feature builders
│   │   ├── nil_features.py     # 57 NIL features
│   │   ├── portal_features.py  # 31 portal features
│   │   └── draft_features.py   # 63 draft features
│   │
│   └── models/                 # ML models
│       ├── nil_valuator.py     # NIL valuation model
│       ├── portal_predictor.py # Flight risk & fit models
│       ├── draft_projector.py  # Draft projection model
│       ├── win_model.py        # Win impact model
│       └── roster_optimizer.py # Budget optimization (PuLP)
│
├── dashboard/                  # Streamlit dashboard
│   ├── app.py                  # Main page
│   ├── pages/                  # Multi-page app
│   │   ├── nil_valuator.py
│   │   ├── portal_intelligence.py
│   │   ├── draft_tracker.py
│   │   └── roster_builder.py
│   └── utils/                  # Dashboard utilities
│
├── tests/                      # Pytest test suite
│   ├── conftest.py             # Fixtures
│   ├── test_models/
│   ├── test_api/
│   └── test_data_collection/
│
├── scripts/
│   └── run_pipeline.py         # End-to-end pipeline
│
├── data/
│   ├── raw/                    # Raw collected data
│   ├── processed/              # Engineered features
│   └── cache/                  # API response cache
│
├── outputs/
│   ├── predictions/            # Model predictions
│   └── reports/                # Generated reports
│
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

---

## Roadmap

### Phase 1: Core Platform (Current)
- [x] NIL Valuator with SHAP explanations
- [x] Portal flight risk prediction
- [x] Draft projections
- [x] Roster optimization
- [x] REST API
- [x] Streamlit dashboard

### Phase 2: Enhanced Intelligence (Q2 2025)
- [ ] Real-time portal alerts (webhook notifications)
- [ ] Automated daily data refresh
- [ ] Historical trend analysis
- [ ] Conference-specific models
- [ ] Improved social media sentiment analysis

### Phase 3: Platform Expansion (Q3 2025)
- [ ] Mobile app (React Native)
- [ ] Full PlaymakerVC integration
- [ ] Cap IQ cross-product features (NFL contract comps)
- [ ] Team subscription portal
- [ ] White-label API for partners

### Phase 4: Advanced Analytics (Q4 2025)
- [ ] Computer vision for game film analysis
- [ ] NLP for media coverage sentiment
- [ ] Real-time game impact predictions
- [ ] Recruiting class projections
- [ ] Championship probability models

---

## Contributing

This is a proprietary project. For contribution guidelines, contact Elite Sports Solutions.

---

## License

All Rights Reserved - Elite Sports Solutions

See [LICENSE](LICENSE) for details.

---

## Support

- **Documentation**: [docs.portaliq.app](https://docs.portaliq.app)
- **API Status**: [status.portaliq.app](https://status.portaliq.app)
- **Email**: support@elitesportssolutions.com
- **Issues**: [GitHub Issues](https://github.com/elitesportssolutions/portal-iq/issues)

---

*Built with precision by Elite Sports Solutions*
