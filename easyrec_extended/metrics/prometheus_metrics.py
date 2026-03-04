"""
Prometheus metrics definitions for EasyRec-Extended.

All metrics are instantiated as module-level singletons so they are shared
across the entire process lifetime.  Import the individual metric objects from
this module to record observations.

Example::

    from easyrec_extended.metrics.prometheus_metrics import recommendation_requests_total
    recommendation_requests_total.labels(status="success", endpoint="/api/v1/recommend").inc()
"""
from prometheus_client import Counter, Gauge, Histogram, Info

# ---------------------------------------------------------------------------
# Request-level counters / histograms
# ---------------------------------------------------------------------------

recommendation_requests_total = Counter(
    "recommendation_requests_total",
    "Total number of recommendation requests",
    ["status", "endpoint"],
)

recommendation_latency_seconds = Histogram(
    "recommendation_latency_seconds",
    "Recommendation pipeline stage latency in seconds",
    ["stage"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ---------------------------------------------------------------------------
# Recall / ranking quality histograms
# ---------------------------------------------------------------------------

recall_items_count = Histogram(
    "recall_items_count",
    "Number of candidate items returned by the recall stage",
    buckets=(0, 5, 10, 20, 50, 100, 200, 500, 1000),
)

ranking_model_score = Histogram(
    "ranking_model_score",
    "Distribution of scores assigned by the ranking model",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ---------------------------------------------------------------------------
# Model inference latency
# ---------------------------------------------------------------------------

model_inference_latency_seconds = Histogram(
    "model_inference_latency_seconds",
    "Model inference latency in seconds",
    ["model_version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# Model version gauges / info
# ---------------------------------------------------------------------------

model_loaded_versions = Gauge(
    "model_loaded_versions",
    "Number of model versions currently loaded",
)

active_model_version = Info(
    "active_model_version",
    "Currently active model version",
)

# ---------------------------------------------------------------------------
# Feature service latency
# ---------------------------------------------------------------------------

feature_service_latency_seconds = Histogram(
    "feature_service_latency_seconds",
    "Feature retrieval latency in seconds",
    ["feature_type"],
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
