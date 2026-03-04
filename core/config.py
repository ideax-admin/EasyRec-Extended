import os
from typing import Dict, Any

class BaseConfig:
    SERVICE_NAME = 'EasyRec-Extended'
    VERSION = '1.0.0'
    DEBUG = False
    DEFAULT_RECALL_SIZE = 100
    DEFAULT_RESULT_SIZE = 20
    MAX_RESULT_SIZE = 100
    POLICY_TIMEOUT_MS = 1000
    FUSION_STRATEGY = 'weighted'
    FUSION_WEIGHTS = {'recall': 0.3, 'collaborative': 0.3, 'content': 0.2, 'popular': 0.2}
    RANKING_MODEL = 'xgboost'
    CACHE_ENABLED = True
    CACHE_TTL = 3600
    CACHE_BACKEND = 'redis'
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME', 'easyrec_db')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    LOG_LEVEL = 'INFO'
    API_HOST = '0.0.0.0'
    API_PORT = 5000

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    CACHE_TTL = 600

class ProductionConfig(BaseConfig):
    DEBUG = False
    LOG_LEVEL = 'INFO'
    SERVICE_REPLICAS = 3

class TestingConfig(BaseConfig):
    DEBUG = True
    CACHE_ENABLED = False
    DB_NAME = 'easyrec_test_db'

def get_config(env: str = None) -> BaseConfig:
    env = env or os.getenv('ENV', 'development')
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    return config_map.get(env, DevelopmentConfig)()