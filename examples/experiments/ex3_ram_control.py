from artiq.experiment import *
from artiq.coredevice.ad9910 import *
from jax import SinaraEnvironment, AD9910RAM, RAMType, RAMProfile, RAMProfileMap
import numpy as np


# __all__ in an experiment module should typically only include the experiment class.
# Specially, it cannot include the base experiment class.
# ARTIQ discovers experiments by trying to load all objects that are subclasses of
# artiq.experiment.Experiment. If __all__ includes the base experiment classs,
# ARTIQ will try to load the base experiment class which results in an error.
__all__ = ["RAM"]


class RAM(JaxExperiment, SinaraEnvironment):
    """Example experiment generating a DDS amplitude ramp waveform, and a frequency sweeo.

    An experiment must first inherit from a base experiment and then inherit an environment.
    The base experiment defines the self.run() function, and interactions with the RAM
    profiles.
    The environment sets up the labrad connection and provides functions for data saving,
    loading parameters, resetting hardware, etc.

    This is an experiment generating a repeatitive DDS amplitude ramp, as well as a
    frequency sweep. To run this experiment, you need to run the "artiq" labrad server.
    Change the DDSes and Urukul CPLDs to use different Urukuls and different RF channels.
    Change the numpy generated array to generate a different waveform.

    Generating with an amplitude RAM profile & a non-amplitude RAM profile introduces
    an approximately 40ns delay.

    Before running this experiment, the DDS output should be terminated with a 50 ohm terminator.
    """
    def build(self):
        super().build()  # Calls JaxExperiment.build(), which calls SinaraEnvironment.build()
        self.cpld = self.get_device("urukul0_cpld")
        self.dds0 = self.get_device("urukul0_ch0")
        self.dds1 = self.get_device("urukul0_ch1")
        self.dds2 = self.get_device("urukul0_ch2")
        self.dds3 = self.get_device("urukul0_ch3")

    def prepare(self):
        super().prepare()  # Calls JaxExperiment.prepare(), which calls SinaraEnvironment.prepare()

        # Generate a linearly growing amplitude, in a list
        # When targeting amplitude in RAM, amplitude modulation is performed
        amp = np.linspace(0.0, 1.0, num=10).tolist()
        freq = np.linspace(1e6, 100e6, num=10).tolist()
        phase = [0.2, 0.7]
        polar = [[0.2, 0.0], [0.7, 0.8]]

        ram_profile0 = RAMProfile(
            self.dds0, amp, 400*ns, RAMType.AMP, RAM_MODE_CONT_RAMPUP,
            base_frequency=100*MHz)
        ram_profile1 = RAMProfile(
            self.dds1, freq, 400*ns, RAMType.FREQ, RAM_MODE_CONT_RAMPUP,
            base_amplitude=1.0)
        ram_profile2 = RAMProfile(
            self.dds2, phase, 400*ns, RAMType.PHASE, RAM_MODE_CONT_RAMPUP,
            base_frequency=100*MHz, base_amplitude=1.0)
        ram_profile3 = RAMProfile(
            self.dds3, polar, 400*ns, RAMType.POLAR, RAM_MODE_CONT_RAMPUP,
            base_frequency=100*MHz, base_amplitude=1.0)

        self.profile_map = RAMProfileMap(self.core)
        self.profile_map.append(self.dds1, ram_profile0)
        self.profile_map.append(self.dds0, ram_profile1)
        self.profile_map.append(self.dds2, ram_profile2)
        self.profile_map.append(self.dds3, ram_profile3)

        # Assign the kernel function here
        self.kernel_func = self.run_kernel

    @kernel
    def run_kernel(self):
        self.core.reset()
        self.core.break_realtime()

        # Initialization of the Urukul and DDS channels
        # A few things must be done to observe the waveform before playing with the RAM:
        #   1. Textbook init(), see ARTIQ API/examples.
        #   2. Turn on the RF switches and give appropriate attenuations.
        self.cpld.init()

        self.init_dds(self.dds0)
        self.init_dds(self.dds1)
        self.init_dds(self.dds2)
        self.init_dds(self.dds3)

        self.profile_map.program()
