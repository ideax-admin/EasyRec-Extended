from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import json

class RecommendationSource(Enum):
    RECALL = 'recall'
    FUSION = 'fusion'
    RANKING = 'ranking'
    BUSINESS_RULE = 'business_rule'

class ItemType(Enum):
    PRODUCT = 'product'
    CONTENT = 'content'
    SERVICE = 'service'
    OTHER = 'other'

@dataclass
class UserContext:
    user_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    device: Optional[str] = None
    location: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    historical_behaviors: List[str] = field(default_factory=list)

@dataclass
class Item:
    item_id: str
    title: str
    category: str
    item_type: ItemType = ItemType.PRODUCT
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[RecommendationSource] = None
    features: Dict[str, float] = field(default_factory=dict)

@dataclass
class RecommendationRequest:
    user_context: UserContext
    candidate_size: int = 100
    result_size: int = 20
    filters: Dict[str, Any] = field(default_factory=dict)
    policies: List[str] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)

@dataclass
class RecommendationResult:
    request_id: str
    items: List[Item]
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyConfig:
    name: str
    stage: str
    enabled: bool = True
    priority: int = 5
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 1000