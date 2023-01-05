from artiq.experiment import *
from artiq.coredevice.ad9910 import *
from jax import JaxExperiment
import numpy as np
from enum import Enum


__all__ = ["AD9910RAM"]


class RAMType(Enum):
    """The type of data in the RAM.

    Enums:
        FREQ: Data are treated as frequencies (Hz).
            The exact frequencies will be played-back.
        PHASE: Data are treated as phases (turns)
            Phase modulation will be performed.
        AMP: Data are treated as amplitudes.
            Amplitude modulation will be performed.
        Polar: Data are treated as amplitudes & phase.
            Polar modulation will be performed.
    """
    FREQ = 1
    PHASE = 2
    AMP = 3
    POLAR = 4


class RAMProfile:
    RAM_SIZE = 1024

    def __init__(self, dds, data, ramp_interval, ram_type, ram_mode,
                 base_frequency=0, base_phase=0, base_amplitude=0):
        """Generate a RAM profile to a DDS channel.
            Include optional parameters based on the "Data Source Priority" of
            AD9910. See the switch case below to find out which optional
            parameter is needed.

        Args:
            dds: The DDS that will playback the RAM profile.
            data: Data (amplitude, phase, frequencies) to be put into the RAM.
            ramp_interval: The time interval between each step of the RAM.
                Note: If possible, keep the interval at a multiple of
                    4*T_sysclk. T_sysclk is generally 1ns.
            ram_type: See the RAMType enum.
            ramp_mode: The playback mode of the RAM. See urukul.py in ARTIQ.
            base_frequency: (Optional) Unmodulated DDS frequency.
            base_phase: (Optional) Unmodulated DDS phase.
            base_amplitude: (Optional) Unmodulated DDS amplitude.

        Raises:
            NotImplementedError: Unsupported RAM types found.
        """
        # Make sure the RAM can hold the entire sequence
        assert (len(data) <= RAMProfile.RAM_SIZE)

        ram = np.zeros((len(data),), dtype=int)
        # Avoid contaminating the passed in data list
        data = data.copy()
        # SPI transaction of AD9910 mandates the transmission of higher
        # order words before lower order words
        # The data is reversed such that the first word shows up first
        data.reverse()

        # Initialize default values for the FTW, POW, and ASF registers
        # This is with accordance to set_mu() in AD9910
        self.ftw, self.pow, self.asf = 0, 0, 0x3fff

        # Encode the FTW, POW, ASF, and raw data en masse
        match ram_type:
            case RAMType.FREQ:
                self.dest = RAM_DEST_FTW
                dds.frequency_to_ram(data, ram)
                self.asf = dds.amplitude_to_asf(base_amplitude)
                self.pow = dds.turns_to_pow(base_phase)
            case RAMType.PHASE:
                self.dest = RAM_DEST_POW
                dds.turns_to_ram(data, ram)
                self.ftw = dds.frequency_to_ftw(base_frequency)
                self.asf = dds.amplitude_to_asf(base_amplitude)
            case RAMType.AMP:
                self.dest = RAM_DEST_ASF
                dds.amplitude_to_ram(data, ram)
                self.ftw = dds.frequency_to_ftw(base_frequency)
                self.pow = dds.turns_to_pow(base_phase)
            case RAMType.POLAR:
                self.dest = RAM_DEST_POWASF
                phase, amp = data
                dds.turns_amplitude_to_ram(phase, amp, ram)
                self.ftw = dds.frequency_to_ftw(base_frequency)
            case _:
                raise NotImplementedError

        self.start_addr = 0
        self.end_addr = len(ram) - 1    # Inclusive

        # Configure ther use of OSK. Note that using different OSK settings for
        # different channels introduces step swiching latencies
        if ram_type == RAMType.AMP or ram_type == RAMType.POLAR:
            # Disable OSK. OSK amplitude as it has a higher priority than RAM.
            self.osk_enable = 0
        else:
            # Enable OSK. There are no other utilizable amplitude data source.
            self.osk_enable = 1

        # Conversion to list to avoid type mismatch in the kernel
        self.ram = ram.tolist()

        # Note: Integer conversion may cause inaccuracy
        self.step = int(ramp_interval * dds.sysclk / 4.0)

        self.ram_mode = ram_mode


class RAMProfileMap:
    def __init__(self, core):
        """Initialize a RAM profile to DDS mapping

        Args:
            core: The ARTIQ Core instance for time control.
        """
        self.ram_profile_map = []
        self.cplds = []
        self.core = core

    def append(self, dds, ram_profile):
        """Append a RAM profile into the builder. In addition, register the
            CPLDs that have DDSes that playback the RAM profile.

        Args:
            dds: The DDS that will playback the RAM profile.
            ram_profile: The RAM profile.
        """
        self.ram_profile_map.append((dds, ram_profile))

        # Add the associlated CPLD to the list if not already there.
        # ARTIQ-Python does not support iterration of a Python builtin set
        # The workaround is to maintain a list, but convert it into set to
        # avoid CPLD duplications.
        cplds = set(self.cplds)
        cplds.add(dds.cpld)
        self.cplds = list(cplds)

    @kernel
    def program(self):
        """Program the RAM profiles to the corresponding AD9910s.

        """
        # Switch to profile 0 for RAM. This is the default profile.
        # Using other profiles for RAM is possible with appropriate parameters
        # for future AD9910 function calls.
        for cpld in self.cplds:
            cpld.set_profile(0)

        for dds, ram_profile in self.ram_profile_map:
            # Datasheets strongly recommands setting ram_enable=0 before
            # writing anything to the RAM profiles
            dds.set_cfr1(ram_enable=0)
            dds.cpld.io_update.pulse_mu(8)

            # Write RAM profile that corresponds to the DDS
            dds.set_profile_ram(
                start=ram_profile.start_addr,
                end=ram_profile.end_addr,
                step=ram_profile.step,
                mode=ram_profile.ram_mode)
            dds.cpld.io_update.pulse_mu(8)

            # Program the RAM, break_realtime to avoid RTIOunderflow
            dds.write_ram(ram_profile.ram)
            self.core.break_realtime()

            # Alternatively, use dds.set()
            dds.set_mu(ftw=ram_profile.ftw,
                       pow_=ram_profile.pow,
                       asf=ram_profile.asf,
                       ram_destination=ram_profile.dest)

        # Go back to single-tone profiles
        # This is to ensure the symmetry of states, so enabling and disabling
        # RAM profiles is logically sound
        for cpld in self.cplds:
            cpld.set_profile(7)
        # Queue in RAM enable operations to each DDS
        for dds, ram_profile in self.ram_profile_map:
            dds.set_cfr1(ram_enable=1,
                         ram_destination=ram_profile.dest,
                         osk_enable=ram_profile.osk_enable)

        # Enable the RAM operations using the same I/O update pulse
        for cpld in self.cplds:
            cpld.io_update.pulse_mu(8)


class AD9910RAM(JaxExperiment):
    """Base class for experiments that uses the AD9910 RAM

    """
    @kernel
    def init_dds(self, dds):
        """Enable the supplied DDS with 6.0 dB attenuation

        """
        self.core.break_realtime()
        dds.init()
        dds.set_att(6.*dB)
        dds.cfg_sw(True)

    def run(self):
        self.kernel_func()
