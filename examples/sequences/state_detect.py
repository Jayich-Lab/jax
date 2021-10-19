from artiq.experiment import *
from jax import Sequence


__all__ = ["StateDetect"]


class StateDetect(Sequence):
    """An example state detection sequence.

    Turns on the cool and repump DDSes, and counts the PMT output.
    """
    kernel_invariants = {
        "_cool_dds", "_repump_dds", "_cool_drift_tracker", "_repump_drift_tracker"
    }

    required_parameters = [
        ("devices", "cool_dds"),
        ("devices", "repump_dds"),
        ("devices", "pmt_edge_counter"),
        ("state_detect", "cool_detuning"),
        ("state_detect", "cool_amplitude"),
        ("state_detect", "cool_drift_tracker"),
        ("state_detect", "repump_detuning"),
        ("state_detect", "repump_amplitude"),
        ("state_detect", "repump_drift_tracker"),
        ("state_detect", "detect_time"),
    ]

    def __init__(self, exp, parameter_group):
        super().__init__(exp, parameter_group)
        self.setup()

    @host_only
    def setup(self):
        """Initializes devices and sets constants.

        Float number calculation is expensive on the device, so they should be calculated
        in host code if possible.
        """
        self._cool_dds = self.exp.get_device(self.p.devices.cool_dds)
        self._repump_dds = self.exp.get_device(self.p.devices.repump_dds)
        # pmt needs to be accessed in other classes.
        self.pmt = self.exp.get_device(self.p.devices.pmt_edge_counter)

        phase = 0.
        self._cool_pow = self._cool_dds.turns_to_pow(phase)
        self._repump_pow = self._repump_dds.turns_to_pow(phase)

        s = self.p.state_detect
        self._cool_asf = self._cool_dds.amplitude_to_asf(s.cool_amplitude)
        self._repump_asf = self._repump_dds.amplitude_to_asf(s.repump_amplitude)

        self._cool_drift_tracker = self.exp.get_drift_tracker(s.cool_drift_tracker)
        self._repump_drift_tracker = self.exp.get_drift_tracker(s.repump_drift_tracker)

        self._detect_time_mu = self.exp.core.seconds_to_mu(s.detect_time)

        cool_frequency = self._cool_drift_tracker.get_frequency_host(d.cool_detuning)
        self._cool_ftw = self._cool_dds.frequency_to_ftw(cool_frequency)
        repump_frequency = self._repump_drift_tracker.get_frequency_host(d.repump_detuning)
        self._repump_ftw = self._repump_dds.frequency_to_ftw(repump_frequency)

    @kernel
    def run(self):
        self._cool_dds.set_mu(self._cool_ftw, self._cool_pow, self._cool_asf)
        self._repump_dds.set_mu(self._repump_ftw, self._repump_pow, self._repump_asf)

        delay_mu(self.exp.rtio_cycle_mu)
        self._cool_dds.sw.on()
        self._repump_dds.sw.on()

        # counts rising edges.
        self.pmt.gate_rising_mu(self._detect_time_mu)
        self._cool_dds.sw.off()
        self._repump_dds.sw.off()

        delay_mu(self.exp.rtio_cycle_mu)
        self._cool_dds.set_mu(self._cool_ftw, self._cool_pow, 0)
        self._repump_dds.set_mu(self._repump_ftw, self._repump_pow, 0)
