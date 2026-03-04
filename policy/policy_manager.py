import logging
from typing import Dict, List, Callable, Any
from enum import Enum
from core.models import PolicyConfig

logger = logging.getLogger(__name__)

class PolicyStage(Enum):
    RECALL = 'recall'
    FUSION = 'fusion'
    RANKING = 'ranking'
    BUSINESS_RULES = 'business_rules'

class PolicyManager:
    def __init__(self):
        self.policies: Dict[PolicyStage, List[PolicyConfig]] = {stage: [] for stage in PolicyStage}
        self.policy_executors: Dict[str, Callable] = {}

    def register_policy(self, policy: PolicyConfig) -> None:
        stage = PolicyStage(policy.stage)
        self.policies[stage].append(policy)
        logger.info(f"Registered policy '{policy.name}'")

    def register_executor(self, policy_name: str, executor: Callable) -> None:
        self.policy_executors[policy_name] = executor

    def execute_policies(self, stage: PolicyStage, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f'Executing policies at stage {stage.value}')
        policies = sorted(self.policies[stage], key=lambda p: p.priority)
        for policy in policies:
            if not policy.enabled or policy.name not in self.policy_executors:
                continue
            try:
                executor = self.policy_executors[policy.name]
                result = executor(context, policy.params)
                context['policy_results'] = context.get('policy_results', {})
                context['policy_results'][policy.name] = result
            except Exception as e:
                logger.error(f"Error executing policy '{policy.name}': {str(e)}")
        return context