# Comprehensive Policy Framework Implementation

class Policy:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def evaluate(self, context):
        results = []
        for rule in self.rules:
            result = rule.evaluate(context)
            results.append(result)
        return results

class Rule:
    def __init__(self, condition, action):
        self.condition = condition
        self.action = action

    def evaluate(self, context):
        if self.condition(context):
            return self.action(context)
        return None

# Example of usage:

if __name__ == '__main__':
    def simple_condition(context):
        return context['value'] > 10

    def action(context):
        return f"Action executed with value: {context['value']}"

    policy = Policy("Example Policy", "An example policy to demonstrate the framework.")
    policy.add_rule(Rule(simple_condition, action))

    context = {'value': 15}
    results = policy.evaluate(context)
    for result in results:
        if result:
            print(result)