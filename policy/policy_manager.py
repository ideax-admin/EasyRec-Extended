class PolicyManager:
    """Manages policies across the recommendation pipeline."""
    
    def __init__(self):
        self.policies = {
            'recall': [],
            'fusion': [],
            'ranking': [],
            'business_rules': []
        }
        self.executors = {}
    
    def register_policy(self, stage, policy_config):
        """Register a policy for a specific stage."""
        if stage not in self.policies:
            raise ValueError(f"Unknown stage: {stage}")
        self.policies[stage].append(policy_config)
    
    def register_executor(self, policy_name, executor_func):
        """Register executor function for a policy."""
        self.executors[policy_name] = executor_func
    
    def execute_stage_policies(self, stage, context, data):
        """Execute all policies for a given stage."""
        if stage not in self.policies:
            raise ValueError(f"Unknown stage: {stage}")
        
        policies = self.policies[stage]
        policies.sort(key=lambda p: p.get('priority', 5))
        
        for policy in policies:
            if not policy.get('enabled', True):
                continue
            
            policy_name = policy['name']
            if policy_name not in self.executors:
                continue
            
            try:
                executor = self.executors[policy_name]
                result = executor(context, data, policy.get('params', {}))
                data = result if result is not None else data
            except Exception as e:
                print(f"Error executing policy {policy_name}: {e}")
        
        return data
    
    def disable_policy(self, policy_name):
        """Disable a policy."""
        for stage_policies in self.policies.values():
            for policy in stage_policies:
                if policy.get('name') == policy_name:
                    policy['enabled'] = False
    
    def enable_policy(self, policy_name):
        """Enable a policy."""
        for stage_policies in self.policies.values():
            for policy in stage_policies:
                if policy.get('name') == policy_name:
                    policy['enabled'] = True