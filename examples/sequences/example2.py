from artiq.experiment import *
from jax import Sequence
from jax.examples.sequences.doppler_cool import DopplerCool
from jax.examples.sequences.state_detect import StateDetect


__all__ = ["Example2"]


class Example2(Sequence):
    """Pulse sequence for the example 2 experiment."""
    kernel_invariants = {}

    required_subsequences = [
        DopplerCool,
        StateDetect
    ]

    def __init__(self, exp, parameter_group, cool_time, wait_time):
        super().__init__(exp, parameter_group)
        self.setup(cool_time, wait_time)

    @host_only
    def setup(self, cool_time, wait_time):
        """Initializes devices and sets constants.

        Float number calculation is expensive on the device, so they should be calculated
        in host code if possible.
        """
        self._doppler_cool = DopplerCool(self.exp, self.p, cool_time)
        self._state_detect = StateDetect(self.exp, self.p)
        self._wait_time_mu = self.exp.core.seconds_to_mu(wait_time)

    @kernel
    def run(self) -> TInt32:
        self._doppler_cool.run()  # runs the Doppler cool sequence.
        delay_mu(self._wait_time_mu)
        self._state_detect.run()  # runs the state detect sequence.
        delay_mu(self._wait_time_mu)
        # collects the counts during state detection.
        # this readout function can be called at any time in the kernel, as long as the RTIO FIFO
        # is not filled. Roughly 100 PMT counts can be stored in the FIFO before readout.
        count = self._state_detect.pmt.fetch_count()
        return count
