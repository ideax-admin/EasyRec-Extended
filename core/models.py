from dataclasses import dataclass
from enum import Enum

class RecommendationSource(Enum):
    USER = 'user'
    SYSTEM = 'system'

class ItemType(Enum):
    PRODUCT = 'product'
    SERVICE = 'service'

@dataclass
class UserContext:
    user_id: str
    preferences: dict

@dataclass
class Item:
    item_id: str
    type: ItemType
    attributes: dict

@dataclass
class RecommendationRequest:
    user_context: UserContext
    items: list[Item]
    source: RecommendationSource

@dataclass
class RecommendationResult:
    request: RecommendationRequest
    recommended_items: list[Item]

@dataclass
class PolicyConfig:
    max_recommendations: int
    filtering_criteria: dict

@dataclass
class FeatureVector:
    features: dict
