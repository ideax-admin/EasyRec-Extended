# EasyRec-Extended

An end-to-end recommendation serving framework that extends [alibaba/EasyRec](https://github.com/alibaba/EasyRec) with a policy-driven online serving layer built in Python and Flask.

## Architecture

```
EasyRec-Extended/
├── easyrec_extended/              # Core package
│   ├── adapters/                  # EasyRec adapter layer
│   │   ├── easyrec_adapter.py     # Wraps EasyRec train/export/eval APIs
│   │   ├── model_inference.py     # TF SavedModel online inference
│   │   └── config_bridge.py      # Protobuf → Python config bridge
│   ├── features/
│   │   └── feature_service.py    # User/item feature retrieval
│   └── model_manager.py          # Multi-version model management
│
├── engine/                        # Recommendation pipeline
│   └── recommendation_engine.py  # 4-stage pipeline orchestrator
│
├── serving/                       # HTTP serving layer
│   ├── api.py                     # Flask REST API routes
│   └── health_check.py            # Health check utilities
│
├── online/                        # Online serving
│   └── serving.py                 # RecommendationServer
│
├── offline/                       # Offline training
│   └── training.py                # EasyRec-backed trainer
│
├── policy/                        # Policy management
│   └── policy_manager.py          # Stage-level policy execution
│
├── core/                          # Core data models and config
│   ├── models.py                  # Data classes
│   └── config.py                  # Configuration
│
├── config/
│   └── pipeline.config.example   # EasyRec pipeline config example
│
├── app.py                         # Flask application entry point
├── requirements.txt               # Python dependencies
└── setup.py                       # Package setup
```

## Pipeline Architecture

The recommendation pipeline runs four sequential stages:

```
Request → Recall → Fusion → Ranking → Business Rules → Response
```

1. **Recall** – Retrieve candidate items (supports EasyRec DSSM/two-tower models or rule-based fallback)
2. **Fusion** – Merge results from multiple recall sources with configurable strategies
3. **Ranking** – Score and rank candidates (supports EasyRec DeepFM/WideAndDeep exported models)
4. **Business Rules** – Apply post-processing filters, boosting, and diversity constraints via pluggable policies

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/ideax-admin/EasyRec-Extended.git
cd EasyRec-Extended

pip install -r requirements.txt
# optional: install EasyRec
pip install git+https://github.com/alibaba/EasyRec.git
```

### 2. Configure

Copy the example pipeline config and customise for your dataset:

```bash
cp config/pipeline.config.example config/pipeline.config
# edit config/pipeline.config
```

Create a `.env` file for environment-specific settings:

```bash
export ENV=development
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### 3. Train a Model (offline)

```python
from offline.training import OfflineTrainer

trainer = OfflineTrainer(pipeline_config_path='config/pipeline.config')
trainer.train(model_dir='experiments/my_model')
trainer.export_model(model_dir='experiments/my_model', export_dir='exports/v1')
```

### 4. Run the Serving Service

```bash
python app.py
```

Or with Docker:

```bash
docker-compose up -d
```

### 5. Get Recommendations

```bash
# REST API
curl -X POST http://localhost:5000/api/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "user123", "result_size": 20}'

# Legacy endpoint
curl "http://localhost:5000/recommend?user_id=user123"

# Health check
curl http://localhost:5000/health
```

## REST API Reference

### POST /api/v1/recommend

Get personalized recommendations.

**Request body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | string | ✅ | User identifier |
| result_size | integer | - | Number of results (default: 20) |
| candidate_size | integer | - | Recall candidate pool size (default: 100) |
| filters | object | - | Category/attribute filters |
| policies | array | - | Active policy names |
| business_rules | array | - | Active business rule names |

**Response:**

```json
{
  "status": "success",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "items": [
    {"item_id": "item_0", "title": "Product 0", "score": 0.99}
  ],
  "processing_time_ms": 12.5
}
```

### GET /api/v1/models

List loaded model versions.

### POST /api/v1/models/reload

Hot-reload a model version.

**Request body (JSON):**

```json
{"version": "v1", "model_dir": "/models/v2_export"}
```

### GET /health

Service health check (checks model and Redis).

## EasyRec Integration

`easyrec_extended/adapters/easyrec_adapter.py` wraps EasyRec's core entry points:

```python
from easyrec_extended.adapters import EasyRecAdapter

adapter = EasyRecAdapter('config/pipeline.config')
adapter.train(model_dir='experiments/model')
adapter.export_model(model_dir='experiments/model', export_dir='exports/v1')
adapter.evaluate_model(model_dir='experiments/model')
```

`easyrec_extended/adapters/model_inference.py` provides online inference from an exported SavedModel:

```python
from easyrec_extended.adapters import EasyRecModelInference

infer = EasyRecModelInference('exports/v1')
result = infer.predict({'user_id': 'u1', 'item_id': 'i1'})
```

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=html
```

## Docker Deployment

```bash
docker-compose up -d
# Services: app (5000), postgres (5432), redis (6379), adminer (8080)
```

## License

MIT License – see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'feat: add my feature'`)
4. Push to the branch and open a Pull Request

Follow PEP 8 style and add tests for new functionality.
