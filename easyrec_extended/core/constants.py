# System Constants and Default Configurations for the Recommendation Service

# Recommendation Service Constants

RECOMMENDATION_SERVICE_VERSION = "1.0.0"
RECOMMENDATION_TIMEOUT_SECONDS = 5
DEFAULT_RECOMMENDATION_COUNT = 10

# Default Configurations

DEFAULT_CONFIG = {
    "service_name": "EasyRecRecommendationService",
    "max_users": 1000,
    "max_items": 10000,
    "cache_enabled": True,
    "cache_timeout_seconds": 60,
    "logging_level": "INFO"
}