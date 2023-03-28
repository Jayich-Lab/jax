from enum import Enum

import numpy as np
from artiq.coredevice.ad9910 import (RAM_DEST_ASF, RAM_DEST_FTW, RAM_DEST_POW,
                                     RAM_DEST_POWASF)
from artiq.experiment import kernel


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
    """A RAM profile of a DDS channel.

    RAM profile is an operation mode that feeds RAM data into the DDS as DDS
    parameter(s) (frequency/phase/amplitude). At an user-defined time interval,
    the DDS fetches a new RAM data and updates the selected parameter. See the
    arguments of the constructor (__init__(..)) to tune the interval, DDS
    parameter and data selection of the RAM playback.

    All RAM profiles in the same AD9910 has access to the same 1024 x 32-bits
    RAM. There are 1024 entries, each entry occupies 32-bits.

    Unlike single-tone profiles, RAM profiles only specifies a subset of
    parameters. The remaining parameters are specified through the FTW/POW/ASF
    registers instead (See args base_frequency, base_phase, base_amplitude).
    Note that unnecessary parameters will be ignored.

    For example, a frequency RAM profile feeds RAM as frequencies. Phase and
    amplitude are specified through POW and ASF registers respectively. These
    registers are controlled by the base_phase and base_amplitude args.

    Reference: AD9910 datasheet (Rev. E), Theory of Operation, Mode Priority,
    Table 5: Data Source Priority

    Note: There are different data latencies between different RAM types.
    Amplitude takes 36 fewer sysclk cycles to reach the DDS output. Refer to
    AD9910 datasheet (Rev. E) p.6, Data Latency.

    Attributes:
        ftw: int, the frequency tuning word for the FTW register.
        pow: int, the phase offset word for the POW register.
        asf: int, the amplitude scale factor for the ASF register.
        dest: int, the RAM data destination.
        start_addr: int, the starting RAM address of the RAM profile.
        end_addr: int, the ending (last) RAM address of the RAM profile.
        osk_enable: int, enable/disable OSK for the RAM profile if 1/0.
        ram: list of int, encoded data for the RAM.
        step: int, the number of sysclk cycles per RAM step.
        ram_mode: int, the playback mode of the RAM.

    Args:
        dds: AD9910, the DDS that will playback the RAM profile.
        data: list of float/2-tuples (elaborated below), the data (amplitude,
            phase, frequencies) to be put into the RAM.
            The type should be "list of float" for frequency/phase/amplitude
            RAM; and list of 2-tuples for polar RAM. The tuple consists of 2
            floats. The first represents phase, and the second represents
            amplitude.
        ramp_interval: float, the time interval between each step of the RAM
            mode playback. Keep the interval at a multiple of 4*T_sysclk
            (4*1 ns).
        ram_type: RAMType, see the RAMType enum.
        ram_mode: int, the playback mode of the RAM. AD9910 allows the
            following RAM playback modes.

            - RAM_MODE_DIRECTSWITCH:
                Only the first data in the RAM profile is fed to the DDS

            The other modes can fetch new RAM data to the DDS. The main
            difference is the behavior once the last data is fetched.

            - RAM_MODE_RAMPUP:
                DDS will not fetch anything after getting the last data.
            - RAM_MODE_BIDIR_RAMP:
                Same as RAM_MODE_RAMPUP. However, an additional ramp down mode
                is supported by setting profile=1.
            - RAM_MODE_CONT_BIDIR_RAMP:
                The RAM profile will ramp down after getting the last data.
                Then ramp up after getting the first data, and so on.
            - RAM_MODE_CONT_RAMPUP:
                The RAM profile will repeat itself after getting the last data
                from the RAM.

            See the following sections on the datasheet regarding the exact
            behavior of these 5 modes.
            - RAM Direct Switch Mode (p.33)
            - RAM Ramp-Up Mode (p.34, Figure 43)
            - RAM Bidirectional Ramp Mode (p.38, Figure 46)
            - RAM Continuous Bidirectional Ramp Mode (p.39, Figure 47)
            - RAM Continuous Recirculate Mode (p.40, Figure 48)
        base_frequency: float, the unmodulated DDS frequency.
            0.0 Hz by default. This argument is optional.
        base_phase: float, the unmodulated DDS phase.
            0.0 turns by default. This argument is optional.
        base_amplitude: float, the unmodulated DDS amplitude.
            1.0 by default. This argument is optional.

    Raises:
        NotImplementedError: Unsupported RAM types found.
    """
    RAM_SIZE = 1024

    def __init__(self, dds, data, ramp_interval, ram_type, ram_mode,
                 base_frequency=0, base_phase=0, base_amplitude=1.0):
        # Make sure the RAM can hold the entire sequence
        assert (len(data) <= RAMProfile.RAM_SIZE)

        ram = np.zeros((len(data),), dtype=np.int32)
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
        if ram_type == RAMType.FREQ:
            self.dest = RAM_DEST_FTW
            dds.frequency_to_ram(data, ram)
            self.asf = dds.amplitude_to_asf(base_amplitude)
            self.pow = dds.turns_to_pow(base_phase)
        elif ram_type == RAMType.PHASE:
            self.dest = RAM_DEST_POW
            dds.turns_to_ram(data, ram)
            self.ftw = dds.frequency_to_ftw(base_frequency)
            self.asf = dds.amplitude_to_asf(base_amplitude)
        elif ram_type == RAMType.AMP:
            self.dest = RAM_DEST_ASF
            dds.amplitude_to_ram(data, ram)
            self.ftw = dds.frequency_to_ftw(base_frequency)
            self.pow = dds.turns_to_pow(base_phase)
        elif ram_type == RAMType.POLAR:
            self.dest = RAM_DEST_POWASF
            # Unpack the list of tuples into tuples
            # Zip it again to convert it into 2 lists
            phase, amp = zip(*data)
            dds.turns_amplitude_to_ram(phase, amp, ram)
            self.ftw = dds.frequency_to_ftw(base_frequency)
        else:
            raise NotImplementedError

        self.start_addr = 0
        self.end_addr = len(ram) - 1    # Inclusive

        # Configure the use of OSK.
        if ram_type == RAMType.AMP or ram_type == RAMType.POLAR:
            # Disable OSK. OSK amplitude as it has a higher priority than RAM.
            self.osk_enable = 0
        else:
            # Enable OSK. There are no other utilizable amplitude data source.
            self.osk_enable = 1

        # Conversion to a list of numpy int32 to avoid type inference.
        self.ram = list(ram)

        self.step = int(ramp_interval * dds.sysclk / 4.0)

        self.ram_mode = ram_mode


class RAMProfileMap:
    """A mapping between RAM profiles and the DDS that performs the playback.

    Args:
        core: Core, the ARTIQ Core instance for time control.
    """
    def __init__(self, core):
        self._ram_profile_map = []
        self._cplds = []
        self._core = core

    def append(self, dds, ram_profile):
        """Append a RAM profile into the builder.

        In addition, register the CPLDs that have DDSes that playback the RAM
        profile.

        Args:
            dds: AD9910, the DDS that will playback the RAM profile.
            ram_profile: RAMProfile, the RAM profile.
        """
        self._ram_profile_map.append((dds, ram_profile))

        # Add the associated CPLD to the list if not already there.
        # ARTIQ-Python does not support iteration of a Python builtin set
        # The workaround is to maintain a list, but convert it into set to
        # avoid CPLD duplications.
        cplds = set(self._cplds)
        cplds.add(dds.cpld)
        self._cplds = list(cplds)

    @kernel
    def load_ram(self):
        """Load RAM operation.

        It initializes the RAM content and profiles.

        Timing: (Figures generated using Kasli 2.0, 1024 RAM entries, 1 DDS)
            The entire load_ram() call is measured to have taken ~620 us
            (omitting the wasted cycles due to the RTIO FIFO being filled up).
            The difference between the RTIO timestamp cursor and the RTIO
            counter (slack) reduced by ~50 us (with the same omission).

            The omission was applied since providing excessive slack will only
            result in wasted clock cycles. load_ram() will hang and waste
            clock cycles if the RTIO FIFO is saturated. No more RTIO events
            could be submitted until some of these events are executed, which
            frees up the FIFO.

            Each RAM entry contributes to the slack loss. From experimentation
            and generalization, the first 127 RAM entries each contributes
            approximately ~60 ns of slack loss on average. The rest of the RAM
            entries each contributes to ~40 ns of slack loss on average.
        """
        # Switch to profile 0 for RAM. This is the default profile.
        # Using other profiles for RAM is possible with appropriate parameters
        # for future AD9910 function calls.
        for cpld in self._cplds:
            cpld.set_profile(0)

        for dds, ram_profile in self._ram_profile_map:
            # The datasheet strongly recommends setting ram_enable=0 before
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

            # Program the RAM, break_realtime to avoid RTIOUnderflow
            dds.write_ram(ram_profile.ram)
            self._core.break_realtime()

            # Alternatively, use dds.set()
            # set_mu()/set() triggers an I/O update to latch the data
            # It wouldn't matter when we change the profile at the end
            dds.set_mu(ftw=ram_profile.ftw,
                       pow_=ram_profile.pow,
                       asf=ram_profile.asf,
                       ram_destination=ram_profile.dest)

        # Go back to single-tone profiles
        # This is to ensure the symmetry of states, so enabling and disabling
        # RAM profiles is logically sound
        for cpld in self._cplds:
            cpld.set_profile(7)

    @kernel
    def enable(self):
        """Set register appropriately for enabling RAM mode.

        After the function is called. RAM mode is NOT YET active.
        Commit RAM enable by calling commit_enable() after.
        """
        # Queue in RAM enable operations to each DDS
        for dds, ram_profile in self._ram_profile_map:
            dds.set_cfr1(ram_enable=1,
                         ram_destination=ram_profile.dest,
                         osk_enable=ram_profile.osk_enable)

    @kernel
    def commit_enable(self):
        """Commit the previous enable() call. The RAM profile will always
            start from the designated first address.
        """
        # Start RAM mode using the same profile switch update.
        # This is achieved by invoking SPI transactions in the same timestamp.
        now = now_mu()
        for cpld in self._cplds:
            at_mu(now)
            cpld.set_profile(0)
        # The timeline advancement of SPI is **naturally restored** here.
        # Note the ordering of at_mu() and set_profile().

    @kernel
    def disable(self):
        """Set register appropriately for disabling RAM mode.

        After the function is called. RAM mode is STILL active. Commit RAM
        disable by calling commit_disable() after.
        """
        for dds, _ in self._ram_profile_map:
            # Disable RAM and OSK.
            # Single-tone profiles use the profile registers for amplitude
            # control. The state of the OSK does not impact the logic here.
            dds.set_cfr1(ram_enable=0, osk_enable=0)

    @kernel
    def commit_disable(self):
        """Commit the previous disable() call.

        After this call has taken effect (in terms of the ARTIQ timeline),
        single-tone profiles are enabled instead.
        """
        # End RAM mode using the same profile switch update.
        # This is achieved by invoking SPI transactions in the same timestamp.
        now = now_mu()
        for cpld in self._cplds:
            at_mu(now)
            cpld.set_profile(7)
        # The timeline advancement of SPI is **naturally restored** here.
        # Note the ordering of at_mu() and set_profile().
