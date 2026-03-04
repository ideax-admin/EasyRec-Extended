"""
Experiment Manager – in-memory A/B experiment registry.

All state is held in plain Python dicts so the implementation is entirely
self-contained with no external dependencies.  The storage layer can be
swapped for Redis by subclassing and overriding the internal storage methods.

Example::

    mgr = ExperimentManager()
    mgr.create_experiment(
        name="ranking_v2_test",
        control_version="v1",
        treatment_version="v2",
        traffic_split=0.2,
    )
    arm = mgr.assign_user("user_42", "ranking_v2_test")  # "control" or "treatment"
    mgr.record_outcome("user_42", "ranking_v2_test", "click_through_rate", 1.0)
    results = mgr.get_experiment_results("ranking_v2_test")
"""
import logging
import statistics
from typing import Any, Dict, List, Optional

from easyrec_extended.experiment.traffic_splitter import TrafficSplitter

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Registry and outcome tracker for A/B experiments.

    Assignments are computed deterministically via :class:`TrafficSplitter`
    so the same user always maps to the same arm without requiring persistent
    assignment storage.
    """

    def __init__(self):
        self._experiments: Dict[str, dict] = {}
        # outcomes[experiment_name][arm][metric_name] = [values…]
        self._outcomes: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self._splitter = TrafficSplitter()

    # ------------------------------------------------------------------
    # Experiment CRUD
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        control_version: str,
        treatment_version: str,
        traffic_split: float = 0.5,
    ) -> dict:
        """Create a new A/B experiment.

        Args:
            name: Unique experiment identifier.
            control_version: Model version label used for the control arm.
            treatment_version: Model version label used for the treatment arm.
            traffic_split: Fraction of traffic (0.0–1.0) routed to treatment.

        Returns:
            The newly created experiment config dict.

        Raises:
            ValueError: If an experiment with this name already exists.
        """
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")

        config = {
            "name": name,
            "control_version": control_version,
            "treatment_version": treatment_version,
            "traffic_split": traffic_split,
            "active": True,
        }
        self._experiments[name] = config
        self._outcomes[name] = {"control": {}, "treatment": {}}
        logger.info(
            "Created experiment '%s' (control=%s treatment=%s split=%.2f)",
            name,
            control_version,
            treatment_version,
            traffic_split,
        )
        return config

    def get_experiment(self, name: str) -> Optional[dict]:
        """Retrieve an experiment config by name.

        Args:
            name: Experiment identifier.

        Returns:
            Experiment config dict, or ``None`` if not found.
        """
        return self._experiments.get(name)

    def list_experiments(self) -> List[dict]:
        """Return a list of all experiment configs.

        Returns:
            List of experiment config dicts.
        """
        return list(self._experiments.values())

    # ------------------------------------------------------------------
    # User assignment
    # ------------------------------------------------------------------

    def assign_user(self, user_id: str, experiment_name: str) -> str:
        """Assign a user to an experiment arm using consistent hashing.

        Args:
            user_id: The user's unique identifier.
            experiment_name: The experiment to assign the user to.

        Returns:
            ``"control"`` or ``"treatment"``.

        Raises:
            ValueError: If the experiment does not exist.
        """
        config = self._experiments.get(experiment_name)
        if config is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        return self._splitter.split(user_id, config)

    def get_model_version_for_user(self, user_id: str, experiment_name: str) -> Optional[str]:
        """Return the model version a user should be served given an experiment.

        Args:
            user_id: The user's unique identifier.
            experiment_name: The experiment name.

        Returns:
            The model version label (control_version or treatment_version),
            or ``None`` if the experiment does not exist.
        """
        config = self._experiments.get(experiment_name)
        if config is None:
            return None
        arm = self.assign_user(user_id, experiment_name)
        version_key = "treatment_version" if arm == "treatment" else "control_version"
        return config.get(version_key)

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        user_id: str,
        experiment_name: str,
        metric_name: str,
        value: float,
    ):
        """Record an outcome metric for a user in an experiment.

        The user's arm is resolved deterministically so repeated calls for
        the same (user_id, experiment_name) always map to the same arm.

        Args:
            user_id: The user's unique identifier.
            experiment_name: The experiment the outcome belongs to.
            metric_name: Name of the metric being tracked.
            value: Numeric metric value.

        Raises:
            ValueError: If the experiment does not exist.
        """
        if experiment_name not in self._outcomes:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        arm = self.assign_user(user_id, experiment_name)
        arm_outcomes = self._outcomes[experiment_name][arm]
        if metric_name not in arm_outcomes:
            arm_outcomes[metric_name] = []
        arm_outcomes[metric_name].append(float(value))
        logger.debug(
            "Recorded outcome: experiment=%s arm=%s user=%s metric=%s value=%s",
            experiment_name,
            arm,
            user_id,
            metric_name,
            value,
        )

    def get_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """Compute summary statistics for control vs treatment arms.

        Args:
            experiment_name: The experiment to summarise.

        Returns:
            Dict with keys ``"control"`` and ``"treatment"``, each containing
            a mapping of metric_name → ``{count, mean, stdev}`` (or ``None``
            values when fewer than 2 samples exist for stdev).

        Raises:
            ValueError: If the experiment does not exist.
        """
        if experiment_name not in self._outcomes:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        results: Dict[str, Any] = {}
        for arm in ("control", "treatment"):
            arm_results = {}
            for metric_name, values in self._outcomes[experiment_name][arm].items():
                n = len(values)
                mean = statistics.mean(values) if n > 0 else None
                stdev = statistics.stdev(values) if n >= 2 else None
                arm_results[metric_name] = {"count": n, "mean": mean, "stdev": stdev}
            results[arm] = arm_results
        return results
