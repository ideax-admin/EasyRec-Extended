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
│   ├── recommendation_engine.py  # 4-stage pipeline orchestrator (parallel recall)
│   └── parallel_executor.py      # Async parallel recall executor
│
├── serving/                       # Dual-protocol serving layer
│   ├── api.py                     # Flask REST API routes
│   ├── grpc_service.py            # gRPC service implementation
│   ├── health_check.py            # Health check utilities
│   ├── recommendation_pb2.py      # Generated protobuf stubs
│   └── recommendation_pb2_grpc.py # Generated gRPC stubs
│
├── protos/                        # Protocol Buffer definitions
│   ├── recommendation.proto       # Service & message definitions
│   └── generate.sh                # Stub generation script
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
├── app.py                         # Flask + optional gRPC entry point
├── requirements.txt               # Python dependencies
└── setup.py                       # Package setup
```

## Pipeline Architecture

The recommendation pipeline runs four sequential stages, with **parallel recall**
across multiple recall engines:

```
Request → Recall (parallel) → Fusion → Ranking → Business Rules → Response
```

1. **Recall** – Retrieve candidate items in parallel from all registered engines
   (timeout-protected; falls back to synthetic items if all engines fail)
2. **Fusion** – Merge results from multiple recall sources with configurable strategies
3. **Ranking** – Score and rank candidates (supports EasyRec DeepFM/WideAndDeep exported
   models; degrades to score-based ranking under timeout pressure)
4. **Business Rules** – Apply post-processing filters, boosting, and diversity constraints
   via pluggable policies

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
# HTTP only (default)
python app.py

# HTTP + gRPC on port 50051
GRPC_ENABLED=true python app.py

# HTTP + gRPC on a custom port
GRPC_ENABLED=true GRPC_PORT=50052 python app.py
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

## gRPC API

EasyRec-Extended can serve recommendations over gRPC for low-latency production use.
The proto definition is in `protos/recommendation.proto`.

### Enable gRPC

```bash
# Via environment variable
GRPC_ENABLED=true GRPC_PORT=50051 python app.py

# Via Make
make serve-grpc
```

### Regenerate Stubs (after editing the proto)

```bash
make proto
# or: bash protos/generate.sh
```

### gRPC Methods

| Method | Description |
|--------|-------------|
| `Recommend` | Get personalized recommendations |
| `HealthCheck` | Check service health |
| `ListModels` | List loaded model versions |
| `ReloadModel` | Hot-reload a model version |

### Example with grpcurl

```bash
# Health check
grpcurl -plaintext localhost:50051 recommendation.RecommendService/HealthCheck

# Get recommendations
grpcurl -plaintext -d '{"user_id":"user123","result_size":10}' \
  localhost:50051 recommendation.RecommendService/Recommend

# List models
grpcurl -plaintext localhost:50051 recommendation.RecommendService/ListModels
```

### Dual-Protocol Architecture

When `GRPC_ENABLED=true`, both protocols share the same `model_manager`,
`server`, and `feature_service` instances:

```
                   ┌───────────────────────────────────┐
HTTP :5000  ──────►│  Flask REST API  (serving/api.py) │
                   │                                   │──► RecommendationServer
gRPC :50051 ──────►│  gRPC Service    (grpc_service.py)│         │
                   └───────────────────────────────────┘    ModelManager
```

### Parallel Recall Configuration

The recall stage runs all registered engines concurrently:

```python
from engine.recommendation_engine import RecommendationEngine

engine = RecommendationEngine(
    recall_timeout_ms=500,    # per-engine timeout (default 500 ms)
    request_timeout_ms=2000,  # overall pipeline timeout (default 2000 ms)
)
engine.register_recall_engine('cf', collaborative_filter_engine)
engine.register_recall_engine('popular', popular_items_engine)
# Both engines run in parallel; slow ones are skipped automatically
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
