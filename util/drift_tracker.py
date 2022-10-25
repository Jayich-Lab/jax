import time as _t
import numpy as _np
import scipy.constants as _c
from artiq.experiment import *
from jax.util.tree_dict import TreeDict


__all__ = ["DriftTracker"]


class ZeemanTracker:
    """Calculates Zeeman shifts.

    Args:
        Zeeman_dict: dict that includes the following key-value pairs:
            lower_gF: float, Lande g-factor of the lower state.
            upper_gF: float, Lande g-factor of the upper state.
            B: float, magnetic field in T.
    """
    kernel_invariants = {
        "lower_gF", "upper_gF"
    }

    def __init__(self, Zeeman_dict):
        self.lower_gF = Zeeman_dict["lower_gF"]
        self.upper_gF = Zeeman_dict["upper_gF"]
        self.B = Zeeman_dict["B"]


class DriftTracker:
    """Converts detuning to DDS frequency.

    Args:
        drift_tracker_dict: dict that includes the following key-value pairs:
            center_frequency: float, frequency of the DDS when the light is resonant.
            detuning_factor: int, factor to convert DDS frequency to detuning.
                Use +/-2 when driving a double-pass AOM, and +/-1 when driving a single-pass AOM.
                The plus(minus) sign is from shifting to higher(lower) frequency.
            center_drift_rate: float, drift rate of center_frequency in Hz/s.
            last_calibration: float, epoch time when the drift tracker was calibrated.
            Zeeman: dict or None, parameters to calculate Zeeman shifts.
                See ZeemanTracker for details. If None, all ZeemanTracker attributes are set to 0.
    """
    kernel_invariants = {
        "detuning_factor", "uB_over_h"
    }

    def __init__(self, drift_tracker_dict):
        self.center_frequency = drift_tracker_dict["center_frequency"]
        self.detuning_factor = drift_tracker_dict["detuning_factor"]
        self.center_drift_rate = drift_tracker_dict["center_drift_rate"]
        self.last_calibration = drift_tracker_dict["last_calibration"]
        if drift_tracker_dict["Zeeman"] is None:
            # define "Zeeman" key so ARTIQ python type checking works when there are instances of
            # DriftTrackers with and without "Zeeman" defined in the dict.
            drift_tracker_dict["Zeeman"] = {
                "upper_gF": 0., "lower_gF": 0., "B": 0.
            }
        self.Zeeman = ZeemanTracker(drift_tracker_dict["Zeeman"])
        self._last_calibration_mu = _np.int64(0)
        self._center_drift_rate_mu = 0.
        self.uB_over_h = _c.physical_constants["Bohr magneton"][0] / _c.h

    @rpc
    def get_epoch_time(self) -> TFloat:
        """Returns the current epoch time in seconds."""
        return _t.time()

    @kernel(flags={"fast-math"})
    def sync_time(self, mu: TFloat = 1e-9):
        """Syncs wall clock time with core device time.

        This function sets the drift rate and calibration time in machine units.
        It needs to be called once before `get_frequency_kernel` or `get_Zeeman_frequency_kernel`
        is used. This function contains a remote procedure call so it may be slow.

        Args:
            time_now: float, epoch time now.
            mu: float, machine unit of time in s. Default 1e-9 (1 ns).
        """
        time_now = self.get_epoch_time()
        time_after_calibration = time_now - self.last_calibration
        self._last_calibration_mu = now_mu() - _np.int64(time_after_calibration / mu)
        # time is converted to machine units, frequency is still in Hz.
        self._center_drift_rate_mu = self.center_drift_rate * mu

    @host_only
    def get_frequency_host(self, detuning):
        center_drift = (_t.time() - self.last_calibration) * self.center_drift_rate
        return self.center_frequency + detuning / self.detuning_factor + center_drift

    @kernel(flags={"fast-math"})
    def get_frequency_kernel(self, detuning):
        center_drift = (now_mu() - self._last_calibration_mu) * self._center_drift_rate_mu
        return self.center_frequency + detuning / self.detuning_factor + center_drift

    @host_only
    def get_Zeeman_frequency_host(self, detuning, lower_mF, upper_mF):
        Zeeman_shift = ((self.Zeeman.upper_gF * upper_mF - self.Zeeman.lower_gF * lower_mF)
                        * self.uB_over_h * self.Zeeman.B)
        return self.get_frequency_host(detuning + Zeeman_shift)

    @kernel(flags={"fast-math"})
    def get_Zeeman_frequency_kernel(self, detuning, lower_mF, upper_mF):
        Zeeman_shift = ((self.Zeeman.upper_gF * upper_mF - self.Zeeman.lower_gF * lower_mF)
                        * self.uB_over_h * self.Zeeman.B)
        return self.get_frequency_kernel(detuning + Zeeman_shift)
