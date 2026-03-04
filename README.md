# EasyRec-Extended

## English Section

### 🎯 Core Features

| Feature | Status | Description |
|---|---|---|
| 4-Stage Pipeline | ✅ Complete | Recall -> Fusion -> Ranking -> Business Rules |
| Multi-Fusion Strategy | ✅ Complete | Weighted, Cascading, Stacking |
| Policy-Driven Framework | ✅ Complete | Flexible policy management system |
| Offline Training | ✅ Complete | Multi-model training & evaluation |
| Automated Testing | ✅ Complete | GitHub Actions CI/CD |
| Containerization | ✅ Complete | Docker + Docker Compose |
| Complete Stack | ✅ Complete | App + DB + Cache + Admin |

### 📊 Project Architecture

```text
EasyRec-Extended/
├── core/                          # Core Module
│   ├── models.py                  ✅ Data Models
│   ├── config.py                  ✅ Configuration Management
│   ├── constants.py               ✅ Constants Definition
│   └── policy_base.py             ✅ Policy Base Framework
│
├── engine/                        # Recommendation Engine
│   ├── recommendation_engine.py   ✅ Main Orchestrator
│   ├── fusion/
│   │   └── fusion_engine.py       ✅ Multi-Source Fusion
│   └── ranking/
│       └── ranking_engine.py      ✅ Ranking Engine
│
├── policy/                        # Policy System
│   ├── __init__.py                ✅ Package Init
│   ├── policy_manager.py          ✅ Policy Management
│   ├── recall_policies.py         ✅ Recall Policies
│   ├── fusion_policies.py         ✅ Fusion Policies
│   ├── ranking_policies.py        ✅ Ranking Policies
│   └── business_policies.py       ✅ Business Rules
│
├── online/                        # Online Serving
│   ├── __init__.py                ✅ Package Init
│   ├── serving.py                 ✅ Recommendation Service
│   ├── api.py                     ✅ REST API Endpoints
│   └── health_check.py            ✅ Health Monitoring
│
├── offline/                       # Offline Training
│   └── training.py                ✅ Model Training Pipeline
│
├── tests/                         # Test Suite
│   ├── test_models.py             ✅ Model Tests
│   ├── test_recommendation_engine.py ✅ Engine Tests
│   ├── test_engines.py            ✅ Fusion/Ranking Tests
│   └── test_policy_manager.py     ✅ Policy Management Tests
│
├── .github/workflows/             # CI/CD Pipeline
│   └── test.yml                   ✅ Automated Testing
│
├── Dockerfile                     ✅ Container Image
├── docker-compose.yml             ✅ Container Orchestration
├── requirements.txt               ✅ Dependencies
├── setup.py                       ✅ Package Setup
└── README.md                      📖 Documentation
```

### 🏗️ Architecture Components

#### 1. Recall Stage

Multiple recall strategies to generate candidate items:

- Embedding-based Recall: Neural network similarity
- Collaborative Filtering: User-based, Item-based
- Hot Items: Popular trending items
- Content-based: Feature similarity

#### 2. Fusion Stage

Combines multiple recall results:

- Weighted Fusion: Configurable weights per strategy
- Cascading Fusion: Sequential combination
- Stacking Fusion: Machine learning based combination
- Deduplication: Smart merging

#### 3. Ranking Stage

Fine-grained ranking of items:

- ML Ranking: XGBoost, LightGBM models
- Multi-Task Learning: CTR, Duration, Engagement
- User Personalization: User profile-based
- Diversity Control: Diversity-aware ranking

#### 4. Business Rules

Post-processing and business logic:

- Filtering: Category, price range, brand
- Boosting: Priority items, strategic promotion
- Diversity: Avoid similar items
- Degradation: Fallback strategies

### 🔧 Core Components

#### Core Module

```python
# core/models.py - Data structures
UserContext               # User session information
Item                      # Recommendation items
RecommendationRequest     # Pipeline input
RecommendationResult      # Pipeline output
PolicyConfig              # Policy configuration

# core/config.py - Configuration
BaseConfig                # Default configuration
DevelopmentConfig         # Development environment
ProductionConfig          # Production environment
TestingConfig             # Testing environment
```

#### Engine Module

```python
# engine/recommendation_engine.py
RecommendationEngine
  ├── recall()            # Multi-source recall
  ├── fusion()            # Result fusion
  ├── ranking()           # Fine-grained ranking
  └── apply_rules()       # Business rules

# engine/fusion/fusion_engine.py
FusionEngine
  ├── weighted_fusion()   # Weighted merge
  ├── cascade_fusion()    # Sequential merge
  └── stack_fusion()      # ML-based merge

# engine/ranking/ranking_engine.py
RankingEngine
  ├── ml_ranking()        # ML model ranking
  ├── diversity_ranking() # Diversity-aware
  └── personalize()       # User personalization
```

#### Policy System

```python
# policy/policy_manager.py
PolicyManager
  ├── register_policy()
  ├── get_policy()
  ├── validate_policy()
  └── apply_policies()

# Supported policy types
├── RecallPolicy          # Multi-source recall
├── FusionPolicy          # Fusion strategies
├── RankingPolicy         # Ranking models
└── BusinessPolicy        # Business rules
```

#### Online Serving

```python
# online/serving.py
RecommendationServer
  ├── initialize()
  ├── get_recommendations()
  ├── health_check()
  └── metrics()

# REST Endpoints
GET  /recommend?user_id=<id>&size=<n>
GET  /health
GET  /metrics
```

### 🚀 Quick Start Guide

#### 1. Installation

```bash
# Clone repository
git clone https://github.com/ideax-admin/EasyRec-Extended.git
cd EasyRec-Extended

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

#### 2. Configuration

```bash
# Create .env file
cp .env.example .env

# Update configuration
export ENV=development
export DB_HOST=localhost
export REDIS_HOST=localhost
```

#### 3. Run Application

```bash
# Using Python
python app.py

# Using Docker
docker-compose up -d

# Using Docker (single container)
docker build -t easyrec-extended .
docker run -p 5000:5000 easyrec-extended
```

#### 4. Test APIs

```bash
# Get recommendations
curl "http://localhost:5000/recommend?user_id=user123&size=20"

# Health check
curl "http://localhost:5000/health"

# Root endpoint
curl "http://localhost:5000/"
```

### 📚 Configuration Guide

#### Policy Configuration

```python
from core.models import PolicyConfig

# Create policy configuration
policy = PolicyConfig(
    name="weighted_fusion",
    stage="fusion",
    enabled=True,
    priority=5,
    params={
        "weights": {
            "recall": 0.3,
            "collaborative": 0.3,
            "content": 0.2,
            "popular": 0.2
        }
    },
    timeout_ms=1000
)
```

#### Application Configuration

```python
from core.config import get_config

# Get configuration based on environment
config = get_config(env='production')

# Key settings
config.SERVICE_NAME            # Service name
config.VERSION                 # Version
config.FUSION_STRATEGY         # Fusion strategy
config.RANKING_MODEL           # Ranking model
config.CACHE_ENABLED           # Enable caching
config.CACHE_TTL               # Cache TTL (seconds)
```

### 🧪 Testing

#### Run Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_recommendation_engine.py -v

# With coverage
pytest tests/ --cov=engine --cov-report=html
```

#### Test Suite

```text
tests/
├── test_models.py                    # Model tests
├── test_recommendation_engine.py     # Engine tests
├── test_engines.py                   # Fusion/Ranking tests
└── test_policy_manager.py            # Policy tests
```

### 🐳 Docker Deployment

#### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Services
├── app           # Main application (port 5000)
├── postgres      # Database (port 5432)
├── redis         # Cache (port 6379)
└── adminer       # DB Admin (port 8080)

# Stop services
docker-compose down
```

#### Docker Build

```bash
# Build image
docker build -t easyrec-extended:latest .

# Run container
docker run -d \
  -p 5000:5000 \
  -e ENV=production \
  -e DB_HOST=db \
  -e REDIS_HOST=redis \
  --name easyrec \
  easyrec-extended:latest
```

### 📊 Monitoring & Observability

#### Prometheus Integration

```yaml
# prometheus.yml configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'easyrec'
    static_configs:
      - targets: ['localhost:5000']
```

#### Metrics Exposed

```text
# Recommendation metrics
easyrec_recommendations_total
easyrec_recommendation_latency_ms
easyrec_recall_items_count
easyrec_ranking_score

# System metrics
http_requests_total
http_request_duration_seconds
python_process_virtual_memory_bytes
```

### 🤝 Contributing Guidelines

#### Fork & Clone

```bash
git clone https://github.com/yourusername/EasyRec-Extended.git
cd EasyRec-Extended
```

#### Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

#### Make Changes

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation

#### Run Tests

```bash
pytest tests/ --cov=engine
```

#### Commit & Push

```bash
git add .
git commit -m "feat: Add your feature description"
git push origin feature/your-feature-name
```

#### Create Pull Request

- Describe changes clearly
- Reference related issues
- Ensure CI/CD passes

### 📝 API Reference

#### GET /recommend

Get personalized recommendations.

Parameters:

- `user_id` (string, required): User identifier
- `size` (integer, optional, default=20): Number of recommendations
- `filters` (object, optional): Category/brand filters
- `policies` (array, optional): Active policies

Response:

```json
{
  "user_id": "user123",
  "items": [
    {
      "item_id": "item001",
      "title": "Product Name",
      "score": 0.95,
      "source": "ranking",
      "category": "Electronics"
    }
  ],
  "processing_time_ms": 45.2
}
```

#### GET /health

Health check endpoint.

Response:

```json
{
  "status": "healthy",
  "service": "EasyRec-Extended",
  "version": "1.0.0",
  "components": {
    "database": "ok",
    "cache": "ok",
    "models": "ok"
  }
}
```

### 🔐 Production Considerations

- Rate Limiting: Implement request throttling
- Authentication: Add API key/OAuth2 authentication
- Logging: Centralized logging (ELK stack)
- Monitoring: Prometheus + Grafana
- Backup: Database replication & backup strategy
- Scaling: Horizontal scaling with load balancer

### 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

### 📞 Support & Contact

- Issues: GitHub Issues
- Email: dev@ideax-business.com
- Documentation: Full Documentation
