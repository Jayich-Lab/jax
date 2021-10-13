from artiq.experiment import *
from jax.utilities.tree_dict import TreeDict


class DriftTracker(TreeDict):
    """Converts detuning to DDS frequency.

    Args:
        drift_tracker_dict: dict that includes the following key-value pairs:
            center_frequency: float, frequency of the DDS when the light is resonant.
            detuning_factor: int, factor to convert DDS frequency to detuning.
                Use +/-2 when driving a double-pass AOM, and +/-1 when driving a single-pass AOM.
                The plus(minus) sign is from shifting to higher(lower) frequency.
    """
    kernel_invariants = {
        "detuning_factor"
    }

    def __init__(self, drift_tracker_dict):
        super().__init__(drift_tracker_dict)

    @portable
    def get_frequency(self, detuning):
        return self.center_frequency + detuning / self.detuning_factor
