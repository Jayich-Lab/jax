from artiq.experiment import *
from jax import Sequence


__all__ = ["DopplerCool"]


class DopplerCool(Sequence):
    """An example Doppler cooling sequence."""
    kernel_invariants = {
        "_cool_dds", "_repump_dds", "_cool_drift_tracker", "_repump_drift_tracker"
    }

    required_parameters = [
        ("devices", "cool_dds"),
        ("devices", "repump_dds"),
        ("doppler_cool", "cool_detuning"),
        ("doppler_cool", "cool_amplitude"),
        ("doppler_cool", "cool_drift_tracker"),
        ("doppler_cool", "repump_detuning"),
        ("doppler_cool", "repump_amplitude"),
        ("doppler_cool", "repump_drift_tracker")
    ]

    def __init__(self, exp, parameter_group, cool_time):
        super().__init__(exp, parameter_group)
        self.setup(cool_time)

    @host_only
    def setup(self, cool_time):
        """Initializes devices and sets constants used in the pulse sequence.

        Float number calculation is expensive on the device, so they should be calculated
        in host code if possible.
        """
        self._cool_dds = self.exp.get_device(self.p.devices.cool_dds)
        self._repump_dds = self.exp.get_device(self.p.devices.repump_dds)

        phase = 0.
        self._cool_pow = self._cool_dds.turns_to_pow(phase)
        self._repump_pow = self._repump_dds.turns_to_pow(phase)

        d = self.p.doppler_cool
        self._cool_asf = self._cool_dds.amplitude_to_asf(d.cool_amplitude)
        self._repump_asf = self._repump_dds.amplitude_to_asf(d.repump_amplitude)

        self._cool_drift_tracker = self.exp.get_drift_tracker(d.cool_drift_tracker)
        self._repump_drift_tracker = self.exp.get_drift_tracker(d.repump_drift_tracker)

        self._cool_time_mu = self.exp.core.seconds_to_mu(cool_time)

        cool_frequency = self._cool_drift_tracker.get_frequency_host(d.cool_detuning)
        self._cool_ftw = self._cool_dds.frequency_to_ftw(cool_frequency)
        repump_frequency = self._repump_drift_tracker.get_frequency_host(d.repump_detuning)
        self._repump_ftw = self._repump_dds.frequency_to_ftw(repump_frequency)

    @kernel
    def run(self):
        # set the values of the DDSes
        self._cool_dds.set_mu(self._cool_ftw, self._cool_pow, self._cool_asf)
        self._repump_dds.set_mu(self._repump_ftw, self._repump_pow, self._repump_asf)

        # wait for a RTIO cycle to reduce likelihood of a collision error (see ARTIQ manual)
        # in a complicated pulse sequence. Then turns on the rf switches of DDSes.
        delay_mu(self.exp.rtio_cycle_mu)
        self._cool_dds.sw.on()
        self._repump_dds.sw.on()

        # wait for cool time before turning off the rf switches.
        delay_mu(self._cool_time_mu)
        self._cool_dds.sw.off()
        self._repump_dds.sw.off()

        # Set amplitudes to 0 to eliminate the DDS signal leakthroughs from the switches.
        delay_mu(self.exp.rtio_cycle_mu)
        self._cool_dds.set_mu(self._cool_ftw, self._cool_pow, 0)
        self._repump_dds.set_mu(self._repump_ftw, self._repump_pow, 0)
