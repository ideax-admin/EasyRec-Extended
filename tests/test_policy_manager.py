import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.policy_manager import PolicyManager


class TestPolicyManager(unittest.TestCase):

    def setUp(self):
        self.policy_manager = PolicyManager()

    def test_add_policy(self):
        policy = {'id': 1, 'name': 'Test Policy'}
        self.policy_manager.add_policy(policy)
        self.assertIn(policy, self.policy_manager._policy_store)

    def test_remove_policy(self):
        policy = {'id': 1, 'name': 'Test Policy'}
        self.policy_manager.add_policy(policy)
        self.policy_manager.remove_policy(1)
        self.assertNotIn(policy, self.policy_manager._policy_store)

    def test_get_policy(self):
        policy = {'id': 1, 'name': 'Test Policy'}
        self.policy_manager.add_policy(policy)
        retrieved_policy = self.policy_manager.get_policy(1)
        self.assertEqual(retrieved_policy, policy)

    def test_get_non_existing_policy(self):
        retrieved_policy = self.policy_manager.get_policy(999)
        self.assertIsNone(retrieved_policy)

    def test_update_policy(self):
        policy = {'id': 1, 'name': 'Test Policy'}
        self.policy_manager.add_policy(policy)
        updated_policy = {'id': 1, 'name': 'Updated Policy'}
        self.policy_manager.update_policy(updated_policy)
        self.assertEqual(self.policy_manager.get_policy(1), updated_policy)


if __name__ == '__main__':
    unittest.main()
