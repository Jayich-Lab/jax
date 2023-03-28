import numpy as np
from artiq.coredevice.ad9910 import *
from artiq.experiment import *

from jax import (DRG, AD9910Manager, DRGType, JaxExperiment, RAMProfile,
                 RAMType, SinaraEnvironment)

# __all__ in an experiment module should typically only include the experiment class.
# Specially, it cannot include the base experiment class.
# ARTIQ discovers experiments by trying to load all objects that are subclasses of
# artiq.experiment.Experiment. If __all__ includes the base experiment classs,
# ARTIQ will try to load the base experiment class which results in an error.
__all__ = ["RAMDRGExample"]


class RAMDRGExample(JaxExperiment, SinaraEnvironment):
    """Example experiment generating several DDS waveform, controlled by the AD9910 DRG and the RAM.

    The following waveform are generated:
    - Channel 0: DRG frequency sweep-up with a bidirectional amplitude RAM. The frequency increases
        from 11 MHz to 36 MHz, with 1 MHz increment every step; The ASF first sweeps-up from 0.1 to
        1.0, then sweeps-down back to 0.1. RAM and DRG update at roughly the same time.
        Note: See Data Latency (Pipeline Latency) Table in the Specification section of the AD9910
        datasheet.
    - Channel 1: Amplitude and frequency identical to Channel 0. Phase offset of the waveform
        alternates from 0 to pi every step.
    - Channel 2: Amplitude sweep-up then sweep down, identical to channel 0 and 1. A pi phase
        offset is gradually injected into the waveform. 10 MHz in frequency.
    - Channel 3: Exact same waveform as Channel 0, but RAM and DRG updates are not aligned.

    To run this experiment, you need to run the "artiq" labrad server.
    Change the DDSes and Urukul CPLDs to use different Urukuls and different RF channels.
    Change the RAM and DRG parameters to generate a different waveform.

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

        # An amplitude bidirectional RAM profile
        amp = np.linspace(0.1, 1.0, num=13).tolist()
        ram_amp_bidir = RAMProfile(self.dds0, amp, 400*ns, RAMType.AMP, RAM_MODE_CONT_BIDIR_RAMP)

        # Phase alternator
        phase = np.linspace(0.0, 6.0, num=13).tolist()
        # Bidirectional polar profile
        polar = list(zip(phase, amp))
        ram_polar_bidir = RAMProfile(self.dds1,
                                     polar, 400*ns, RAMType.POLAR, RAM_MODE_CONT_BIDIR_RAMP)

        # Linear frequency ramp-up from 11 MHz to 36 MHz.
        drg_freq = DRG(self.dds0, 11*MHz, 36*MHz, 400*ns, DRGType.FREQ, num_of_steps=25)

        # Number of updates = DRG duration / update period
        drg_phase = DRG(self.dds2, 0.5 / (10000 / 4), 0.5, 4*ns, DRGType.PHASE,
                        num_of_steps=10000//4)

        # Another amplitude bidirectional RAM profile, but with higher DRG update frequency.
        finer_amp = np.linspace(0.1, 1.0, num=16).tolist()
        ram_finer_amp_bidir = RAMProfile(self.dds3,
                                         finer_amp, 320*ns, RAMType.AMP, RAM_MODE_CONT_BIDIR_RAMP)

        self.dds_manager = AD9910Manager(self.core)

        # Bidirectional amplitude with frequency sweep-up.
        self.dds_manager.append(self.dds0, frequency_src=drg_freq, amplitude_src=ram_amp_bidir)
        # Same as channel 0, except the phase is shifted by an extra pi every step.
        # Note the abrupt waveform direction changes between each step.
        self.dds_manager.append(self.dds1, frequency_src=drg_freq,
                                amplitude_src=ram_polar_bidir, phase_src=ram_polar_bidir)
        # Amplitude modulation same as channel 0, with a pi phase shift gradually injected.
        # Observe the waveform at the beginning and the end are opposite in sign. (pi phase offset)
        self.dds_manager.append(self.dds2, frequency_src=10*MHz,
                                amplitude_src=ram_amp_bidir, phase_src=drg_phase)
        # Same as channel 0, with unaligned RAM and DRG updates.
        self.dds_manager.append(self.dds3, frequency_src=drg_freq,
                                amplitude_src=ram_finer_amp_bidir)

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
        # 1. Profile 0 waveform for 10 us
        # 2. Single-tone profile 7 for 10 us
        # 3. Profile 0 waveform for another 10 us
        # 4. Single-tone profile 7 until reset

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
