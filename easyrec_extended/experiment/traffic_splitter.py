"""
Traffic splitter for A/B experiments using consistent hashing.

The same user_id always maps to the same experiment arm (control/treatment)
regardless of when the split is evaluated, enabling reproducible experiments
without persistent assignment storage.
"""
import hashlib
import logging

logger = logging.getLogger(__name__)


class TrafficSplitter:
    """Assigns users to experiment arms using a consistent MD5-based hash.

    Example::

        splitter = TrafficSplitter()
        arm = splitter.split("user123", {"traffic_split": 0.5})
        # Returns "control" or "treatment" deterministically for user123
    """

    def split(self, user_id: str, experiment_config: dict) -> str:
        """Assign a user to "control" or "treatment".

        Uses a deterministic MD5 hash of ``<user_id>:<experiment_name>`` so
        that the same user receives the same arm on every call.

        Args:
            user_id: The user's unique identifier.
            experiment_config: Experiment configuration dict containing at
                minimum:
                - ``name`` (str): experiment name used to salt the hash.
                - ``traffic_split`` (float): fraction of traffic (0.0–1.0)
                  to send to the *treatment* arm.

        Returns:
            ``"treatment"`` when the user falls within the treatment bucket,
            ``"control"`` otherwise.
        """
        experiment_name = experiment_config.get("name", "")
        traffic_split = float(experiment_config.get("traffic_split", 0.5))

        # Clamp to valid range
        traffic_split = max(0.0, min(1.0, traffic_split))

        hash_input = f"{user_id}:{experiment_name}".encode("utf-8")
        digest = hashlib.md5(hash_input).hexdigest()  # nosec B324 – not for security
        # Convert first 8 hex chars to an integer bucket in [0, 1)
        bucket = int(digest[:8], 16) / 0xFFFFFFFF

        arm = "treatment" if bucket < traffic_split else "control"
        logger.debug(
            "TrafficSplitter: user=%s experiment=%s bucket=%.4f split=%.2f arm=%s",
            user_id,
            experiment_name,
            bucket,
            traffic_split,
            arm,
        )
        return arm
