from artiq.coredevice.ad9910 import *
from artiq.experiment import *

from jax import DRG, AD9910Manager, DRGType, JaxExperiment, SinaraEnvironment

# __all__ in an experiment module should typically only include the experiment class.
# Specially, it cannot include the base experiment class.
# ARTIQ discovers experiments by trying to load all objects that are subclasses of
# artiq.experiment.Experiment. If __all__ includes the base experiment classs,
# ARTIQ will try to load the base experiment class which results in an error.
__all__ = ["DRGExample"]


class DRGExample(JaxExperiment, SinaraEnvironment):
    """Example experiment generating several DDS waveform, controlled by the AD9910 DRG.

    The following waveform are generated:
    - Channel 0: Frequency sweep-up. Increasing DDS frequency from 1 MHz to 10 MHz. The frequency
        increments 1 MHz every step. The frequency will stay at 10 MHz after reaching it.
    - Channel 1: Amplitude sweep-up. Increasing DDS amplitude scaling factor (ASF) from 0.01 to 1.0.
        The ASF increments by 0.01 every step. The ASF will reset to 0 after reaching 1.0.
    - Channel 2: Amplitude sweep-up. Increasing DDS amplitude scaling factor (ASF) from 0.01 to
        0.99. The ASF increments by 0.01 every step. The ASF will stay at 0.99. The only difference
        to Channel 1 is the change of `end` and `num_of_steps`.
    - Channel 3: Phase sweep-up. The phase offset linearly increases from 0 to pi.

    To run this experiment, you need to run the "artiq" labrad server.
    Change the DDSes and Urukul CPLDs to use different Urukuls and different RF channels.
    Change the DRG parameters to generate a different waveform.

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

        # The DRG waveform instances.
        # Specify step_gap="fine" and unspecify `num_of_steps` to minimize the step gap.
        drg_freq = DRG(self.dds0, 1*MHz, 10*MHz, 400*ns, DRGType.FREQ, num_of_steps=10)
        drg_amp_no_dwell_high = DRG(self.dds1, 0.01, 1.0, 80*ns, DRGType.AMP,
                                    num_of_steps=100, dwell_high=False)
        # Note: See dwell_high default value.
        drg_amp_dwell_high = DRG(self.dds2, 0.01, 0.99, 80*ns, DRGType.AMP, num_of_steps=99)

        # Number of updates = DRG duration / update period
        drg_phase = DRG(self.dds3, 0.5 / (10000 / 4), 0.5, 4*ns, DRGType.PHASE,
                        num_of_steps=10000//4)

        self.dds_manager = AD9910Manager(self.core)

        # Linear frequency sweep-up.
        self.dds_manager.append(self.dds0, frequency_src=drg_freq, amplitude_src=0.5)
        self.dds_manager.append(self.dds1, frequency_src=40*MHz,
                                amplitude_src=drg_amp_no_dwell_high)
        # To avoid dropping to `start` - "step gap", make sure the difference between
        # `end` + "step gap" will not overflow in reach `start` - "step gap". Note that the
        # verification is conducted in machine unit.
        self.dds_manager.append(self.dds2, frequency_src=60*MHz,
                                amplitude_src=drg_amp_dwell_high)
        # A pi phase shift is injected into the waveform gradually.
        # Observe the phase of the waveform compared to the beginning on the oscilloscope.
        # The actual synthesized frequency should be 1.05 MHz instead of the nominal 1 MHz.
        self.dds_manager.append(self.dds3, frequency_src=1*MHz, amplitude_src=0.7,
                                phase_src=drg_phase)

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
        # 1. DRG waveform for 10 us
        # 2. Single-tone profile 7 for 10 us
        # 3. DRG waveform for another 10 us
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
