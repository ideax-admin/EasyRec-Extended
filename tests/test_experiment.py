"""Tests for A/B experiment framework."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyrec_extended.experiment.traffic_splitter import TrafficSplitter
from easyrec_extended.experiment.experiment_manager import ExperimentManager


class TestTrafficSplitter(unittest.TestCase):
    """Tests for TrafficSplitter."""

    def setUp(self):
        self.splitter = TrafficSplitter()

    def test_split_returns_control_or_treatment(self):
        """split() always returns 'control' or 'treatment'."""
        config = {'name': 'exp1', 'traffic_split': 0.5}
        result = self.splitter.split('user_1', config)
        self.assertIn(result, ('control', 'treatment'))

    def test_split_is_deterministic(self):
        """The same user always gets the same arm."""
        config = {'name': 'exp1', 'traffic_split': 0.5}
        arm1 = self.splitter.split('user_42', config)
        arm2 = self.splitter.split('user_42', config)
        self.assertEqual(arm1, arm2)

    def test_split_different_users_can_differ(self):
        """Different users can be assigned to different arms."""
        config = {'name': 'exp1', 'traffic_split': 0.5}
        arms = {self.splitter.split(f'user_{i}', config) for i in range(100)}
        # With 100 users and 50% split we should see both arms
        self.assertEqual(arms, {'control', 'treatment'})

    def test_split_zero_traffic_always_control(self):
        """0% treatment traffic means all users get control."""
        config = {'name': 'exp_zero', 'traffic_split': 0.0}
        for i in range(20):
            arm = self.splitter.split(f'user_{i}', config)
            self.assertEqual(arm, 'control')

    def test_split_full_traffic_always_treatment(self):
        """100% treatment traffic means all users get treatment."""
        config = {'name': 'exp_full', 'traffic_split': 1.0}
        for i in range(20):
            arm = self.splitter.split(f'user_{i}', config)
            self.assertEqual(arm, 'treatment')

    def test_split_respects_traffic_percentage(self):
        """Empirical split ratio should be close to configured traffic_split."""
        config = {'name': 'exp_stat', 'traffic_split': 0.3}
        n = 500
        treatment_count = sum(
            1 for i in range(n)
            if self.splitter.split(f'user_{i}', config) == 'treatment'
        )
        ratio = treatment_count / n
        # Allow ±10% tolerance
        self.assertAlmostEqual(ratio, 0.3, delta=0.1)


class TestExperimentManager(unittest.TestCase):
    """Tests for ExperimentManager."""

    def setUp(self):
        self.mgr = ExperimentManager()

    def test_create_experiment(self):
        """create_experiment() returns the created config."""
        config = self.mgr.create_experiment(
            name='exp1',
            control_version='v1',
            treatment_version='v2',
            traffic_split=0.5,
        )
        self.assertEqual(config['name'], 'exp1')
        self.assertEqual(config['control_version'], 'v1')
        self.assertEqual(config['treatment_version'], 'v2')
        self.assertEqual(config['traffic_split'], 0.5)

    def test_create_duplicate_raises(self):
        """Creating an experiment with an existing name raises ValueError."""
        self.mgr.create_experiment('dup', 'v1', 'v2')
        with self.assertRaises(ValueError):
            self.mgr.create_experiment('dup', 'v1', 'v2')

    def test_list_experiments_empty(self):
        """list_experiments() returns empty list initially."""
        self.assertEqual(self.mgr.list_experiments(), [])

    def test_list_experiments_returns_all(self):
        """list_experiments() returns all created experiments."""
        self.mgr.create_experiment('e1', 'v1', 'v2')
        self.mgr.create_experiment('e2', 'v1', 'v2')
        names = [e['name'] for e in self.mgr.list_experiments()]
        self.assertIn('e1', names)
        self.assertIn('e2', names)

    def test_get_experiment_returns_none_for_missing(self):
        """get_experiment() returns None for unknown name."""
        self.assertIsNone(self.mgr.get_experiment('missing'))

    def test_get_experiment_returns_config(self):
        """get_experiment() returns the config dict for known experiment."""
        self.mgr.create_experiment('e1', 'v1', 'v2')
        config = self.mgr.get_experiment('e1')
        self.assertIsNotNone(config)
        self.assertEqual(config['name'], 'e1')

    def test_assign_user_returns_arm(self):
        """assign_user() returns 'control' or 'treatment'."""
        self.mgr.create_experiment('e1', 'v1', 'v2')
        arm = self.mgr.assign_user('user_1', 'e1')
        self.assertIn(arm, ('control', 'treatment'))

    def test_assign_user_consistent(self):
        """assign_user() returns the same arm for the same user."""
        self.mgr.create_experiment('e1', 'v1', 'v2')
        arm1 = self.mgr.assign_user('user_42', 'e1')
        arm2 = self.mgr.assign_user('user_42', 'e1')
        self.assertEqual(arm1, arm2)

    def test_assign_user_raises_for_unknown_experiment(self):
        """assign_user() raises ValueError for unknown experiment."""
        with self.assertRaises(ValueError):
            self.mgr.assign_user('user_1', 'does_not_exist')

    def test_get_model_version_for_user_returns_version(self):
        """get_model_version_for_user() returns a version string."""
        self.mgr.create_experiment('e1', 'ctrl_v1', 'treat_v2', traffic_split=0.5)
        version = self.mgr.get_model_version_for_user('user_1', 'e1')
        self.assertIn(version, ('ctrl_v1', 'treat_v2'))

    def test_record_outcome(self):
        """record_outcome() stores metric values without error."""
        self.mgr.create_experiment('e1', 'v1', 'v2')
        self.mgr.record_outcome('user_1', 'e1', 'ctr', 1.0)
        self.mgr.record_outcome('user_2', 'e1', 'ctr', 0.0)

    def test_record_outcome_raises_for_unknown_experiment(self):
        """record_outcome() raises ValueError for unknown experiment."""
        with self.assertRaises(ValueError):
            self.mgr.record_outcome('user_1', 'does_not_exist', 'ctr', 1.0)

    def test_get_experiment_results_structure(self):
        """get_experiment_results() returns control/treatment keys."""
        self.mgr.create_experiment('e1', 'v1', 'v2')
        results = self.mgr.get_experiment_results('e1')
        self.assertIn('control', results)
        self.assertIn('treatment', results)

    def test_get_experiment_results_with_outcomes(self):
        """get_experiment_results() includes mean and count for recorded outcomes."""
        self.mgr.create_experiment('e_res', 'v1', 'v2', traffic_split=1.0)
        # With traffic_split=1.0 all users go to treatment
        for i in range(5):
            self.mgr.record_outcome(f'user_{i}', 'e_res', 'score', float(i))
        results = self.mgr.get_experiment_results('e_res')
        treatment = results['treatment']
        self.assertIn('score', treatment)
        self.assertEqual(treatment['score']['count'], 5)
        self.assertIsNotNone(treatment['score']['mean'])

    def test_get_experiment_results_raises_for_unknown(self):
        """get_experiment_results() raises ValueError for unknown experiment."""
        with self.assertRaises(ValueError):
            self.mgr.get_experiment_results('no_such_experiment')


if __name__ == '__main__':
    unittest.main()
