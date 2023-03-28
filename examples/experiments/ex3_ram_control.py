import numpy as np
from artiq.coredevice.ad9910 import *
from artiq.experiment import *

from jax import (AD9910Manager, JaxExperiment, RAMProfile, RAMType,
                 SinaraEnvironment)

# __all__ in an experiment module should typically only include the experiment class.
# Specially, it cannot include the base experiment class.
# ARTIQ discovers experiments by trying to load all objects that are subclasses of
# artiq.experiment.Experiment. If __all__ includes the base experiment classs,
# ARTIQ will try to load the base experiment class which results in an error.
__all__ = ["RAM"]


class RAM(JaxExperiment, SinaraEnvironment):
    """Example experiment generating several DDS waveform, controlled by the
    AD9910 RAM.

    The following repetitive waveform are generated:
    - Channel 0: Amplitude modulation. The DDS amplitude scale factor increments by 0.1 at
        every step. It increases from 0.1 to 1.0.
    - Channel 1: Frequency sweep. Increasing DDS frequency from 1 MHz to 10 MHz. The frequency
        increments 1 MHz every step.
    - Channel 2: Binary phase modulation. The phase is flipped by 0.5 turns at every step.
    - Channel 3: Polar modulation. The amplitude is the same as channel 0, and the phase is
        flipped by 0.5 turns at every step, just like Channel 2.

    To run this experiment, you need to run the "artiq" labrad server.
    Change the DDSes and Urukul CPLDs to use different Urukuls and different RF channels.
    Change the numpy generated array to generate a different waveform.

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
        amp = np.linspace(0.1, 1.0, num=10).tolist()
        freq = np.linspace(1e6, 10e6, num=10).tolist()
        phase = np.linspace(0.0, 4.5, num=10).tolist()
        polar = list(zip(phase, amp))

        ram_profile0 = RAMProfile(
            self.dds0, amp, 400*ns, RAMType.AMP, RAM_MODE_CONT_RAMPUP)
        ram_profile1 = RAMProfile(
            self.dds1, freq, 400*ns, RAMType.FREQ, RAM_MODE_CONT_RAMPUP)
        ram_profile2 = RAMProfile(
            self.dds2, phase, 400*ns, RAMType.PHASE, RAM_MODE_CONT_RAMPUP)
        ram_profile3 = RAMProfile(
            self.dds3, polar, 400*ns, RAMType.POLAR, RAM_MODE_CONT_RAMPUP)

        self.dds_manager = AD9910Manager(self.core)
        self.dds_manager.append(
            self.dds0, frequency_src=100*MHz, amplitude_src=ram_profile0)
        self.dds_manager.append(
            self.dds1, frequency_src=ram_profile1, amplitude_src=1.0)
        self.dds_manager.append(
            self.dds2, frequency_src=100*MHz, phase_src=ram_profile2, amplitude_src=1.0)
        self.dds_manager.append(
            self.dds3, frequency_src=100*MHz, phase_src=ram_profile3, amplitude_src=ram_profile3)

    @kernel
    def init_dds(self, dds):
        """Enable the supplied DDS with 6.0 dB attenuation

        """
        self.core.break_realtime()
        dds.init()
        dds.set_att(6.*dB)
        dds.cfg_sw(True)

    @kernel
    def run(self):
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

        # Prepare a RAM profile & a single-tone profile
        for dds in [self.dds0, self.dds1, self.dds2, self.dds3]:
            dds.set(frequency=5*MHz, amplitude=0.2)

        self.dds_manager.load_profile()

        # DDS output sequence:
        # 1. RAM profiles for 10 us
        # 2. Single-tone profiles for 10 us
        # 3. RAM profiles for another 10 us
        # 4. Single-tone profiles until reset

        self.dds_manager.enable()
        # Record time right before commit
        now = now_mu()
        self.dds_manager.commit_enable()
        self.dds_manager.disable()

        now += self.core.seconds_to_mu(10*us)
        at_mu(now)
        self.dds_manager.commit_disable()
        self.dds_manager.enable()

        now += self.core.seconds_to_mu(10*us)
        at_mu(now)
        self.dds_manager.commit_enable()
        self.dds_manager.disable()

        now += self.core.seconds_to_mu(10*us)
        at_mu(now)
        self.dds_manager.commit_disable()
