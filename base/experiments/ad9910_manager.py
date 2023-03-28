from artiq.coredevice import ad9910
from artiq.experiment import kernel

from jax import DRG, DRGMap, DRGType, RAMProfile, RAMProfileMap, RAMType


class AD9910Manager:
    """A Manager class that manages the real-time operation of a specific AD9910 profile.

    It supports both RAM and single-tone profiles, and may optionally include a DRG. This sill
    always on profile 0.

    Args:
        core: Core, the ARTIQ Core instance for time control.
    """
    def __init__(self, core):
        self._drg_map = DRGMap()
        self._ram_profile_map = RAMProfileMap(core)
        self._cfg_map = []
        self._cplds = []
        # Function pointers. Workaround for the zero-length list type inference issue.
        self._load_ram_profiles_fp = self.bypass
        self._load_drg_fp = self.bypass

    class DDSConfig:
        """A collection of basic AD9910 configurations, that does not belong to DRG or RAM.

        Args:
            dds: AD9910, the DDS that will playback the profile.
            drg: DRG, the digital ramp generator configuration. Defaults to None.
            ram_profile: RAMProfile, the RAM profile. Defaults to None.
            frequency: float, the baseline DDS frequency in the FTW register. Defaults to 0.0.
            phase: float, the baseline DDS phase offset in the POW register. Defaults to 0.0.
            amplitude: float, the baseline DDS amplitude scale factor in the ASF register.
                Defaults to 1.0.
        """
        def __init__(self, dds,
                     drg=None, ram_profile=None, frequency=0.0, phase=0.0, amplitude=1.0):
            self.ftw = dds.frequency_to_ftw(frequency)
            self.pow_ = dds.turns_to_pow(phase)
            self.asf = dds.amplitude_to_asf(amplitude)

            if ram_profile is not None:
                self.ram_destination = ram_profile.dest
                self.ram_enable = 1
            else:
                self.ram_destination = 0
                self.ram_enable = 0

            if drg is not None:
                self.drg_destination = drg.dest
                self.drg_enable = 1
            else:
                self.drg_destination = 0
                self.drg_enable = 0

            self.osk_enable = 1
            # OSK is never needed if RAM is disabled.
            if not self.ram_enable:
                self.osk_enable = 0
            # Disable OSK to avoid overriding RAM with amplitude destination
            elif ram_profile.ram_type in [RAMType.AMP, RAMType.POLAR]:
                self.osk_enable = 0
            # Disable OSK to avoid overriding amplitude DRG
            elif (drg is not None) and drg.drg_type == DRGType.AMP:
                self.osk_enable = 0

    def append(self, dds, frequency_src=0.0, phase_src=0.0, amplitude_src=1.0):
        """Append a AD9910 profile.

        An AD9910 profile may or may not use the RAM or the DRG. If RAM profile is not specified,
        a single-tone profile will be configured instead.

        The DDS settings supplied from the parameters will be always loaded to the profile 0 of the
        specified DDS. This includes everything about the DDS, e.g. using RAM profile, DRG,
        configuration register values, etc.).

        Args:
            dds: AD9910, the DDS that will playback the profile.
            frequency_src: DRG/RAMProfile/float, the source of the DDS frequency. It can be a DRG,
                RAMProfile, or a simple constant that specifies a fixed frequency. Defaults to 0.0.
            phase_src: float, DRG/RAMProfile/float, the source of the DDS phase offset. It can be a
                DRG, RAMProfile, or a simple constant that specifies a fixed phase offset.
                Defaults to 0.0.
            amplitude_src: float, DRG/RAMProfile/float, the source of the DDS amplitude scale
                factor. It can be a DRG, RAMProfile, or just a simple constant. that specifies a
                fixed amplitude scaling factor. Defaults to 1.0.

        Raises:
            ValueError: Invalid DDS parameter source specified.
        """
        sources = (frequency_src, phase_src, amplitude_src)
        src_names = ("frequency", "phase", "amplitude")

        # Validate arguments. DRG/RAMProfile must have appropriate destinations individually.
        drg_types_dict = {
            "frequency": [DRGType.FREQ],
            "phase": [DRGType.PHASE],
            "amplitude": [DRGType.AMP],
        }
        ram_types_dict = {
            "frequency": [RAMType.FREQ],
            "phase": [RAMType.PHASE, RAMType.POLAR],
            "amplitude": [RAMType.AMP, RAMType.POLAR],
        }

        for src, name in zip(sources, src_names):
            if (type(src) == DRG) and (src.drg_type not in drg_types_dict[name]):
                raise ValueError("Invalid DRG data destination")
            if (type(src) == RAMProfile) and (src.ram_type not in ram_types_dict[name]):
                raise ValueError("Invalid RAM data destination")

        # Check that the DDS uses at most 1 DRG, 1 RAM Profile
        drgs = [src for src in sources if type(src) == DRG]
        rams = [src for src in sources if type(src) == RAMProfile]

        num_of_drg = len(drgs)
        num_of_ram = len(rams)

        if num_of_drg > 1:
            raise ValueError("AD9910 configured with multiple DRGs")
        if num_of_ram > 1:
            # Edge case: RAMProfile can specify amplitude + phase (POLAR)
            # Group everything into one check would be simpler and easier to read.
            # However, it wouldn't precisely convey the error to the user.
            if num_of_ram > 2 or (num_of_ram >= 2 and type(frequency_src) == RAMProfile):
                raise ValueError("AD9910 configured with multiple RAM Profiles")
            # Now, we are certain that the RAM profile targets both amplitude and phase.
            # POLAR RAM must be configured.
            if phase_src.ram_type != RAMType.POLAR or amplitude_src.ram_type != RAMType.POLAR:
                raise ValueError("Only POLAR RAM allow both amplitude & phase as data destinations")
            # Finally, we should also make sure that the RAM are the same object
            if phase_src is not amplitude_src:
                raise ValueError("Found 2 different POLAR RAMs")

        # Now, we are certain that the DRG and RAM list contains a single valid element
        # Note: Naive implementation: Use set().
        # Issue of set(): We cannot validate that correct positioning of the arguments.
        arg_dict = {}
        for src, name in zip(sources, src_names):
            if type(src) == DRG:
                arg_dict["drg"] = src
            elif type(src) == RAMProfile:
                # Duplicate entry in dictionary will be overwritten.
                # By this point. We are certain that the duplicated value is the same object.
                arg_dict["ram_profile"] = src
            else:
                arg_dict[name] = src
        cfg = self.DDSConfig(dds, **arg_dict)

        # Finally, store the RAM profiles and DRG config.
        # Dynamically linking methods workarounds the zero-length array type inference issue.
        # If there isn't code for the compiler to compile, there won't be any bugs!
        drg = arg_dict.get("drg")
        ram_profile = arg_dict.get("ram_profile")
        if drg is not None:
            self._drg_map.append(dds, drg)
            self._load_drg_fp = self._drg_map.load
        if ram_profile is not None:
            self._ram_profile_map.append(dds, ram_profile)
            self._load_ram_profiles_fp = self._ram_profile_map.load_ram
        self._cfg_map.append((dds, cfg))

        cplds = set(self._cplds)
        cplds.add(dds.cpld)
        self._cplds = list(cplds)

    @kernel
    def bypass(self):
        """It does nothing!

        However, it is critical that this function is a kernel method, so the compiler can optimize
        it (very likely by omission). Leaving it on RPC will have a significant impact on timing.
        """
        pass

    @kernel
    def load_profile(self):
        """Load all configurations for single-tone/RAM profile and DRG.
        """
        self._load_ram_profiles_fp()
        self._load_drg_fp()

        # Configure as RAM profile if and only if a RAMProfile was submitted.
        for dds, cfg in self._cfg_map:
            if not cfg.ram_enable:
                dds.set_mu(ftw=cfg.ftw, pow_=cfg.pow_, asf=cfg.asf, profile=0)
            else:
                dds.set_mu(ftw=cfg.ftw, pow_=cfg.pow_, asf=cfg.asf,
                           ram_destination=cfg.ram_destination)

    @kernel
    def enable(self):
        """Set register appropriately for enabling profile 0.

        After the function is called. Both RAM mode and the DRG are NOT active. Commit profile
        enable by calling commit_enable() after.
        """
        # Relevant AD9910 CFR2 register flags.
        ENABLE_SINGLE_TONE_ASF = 24
        DRG_DEST = 20
        DRG_ENABLE = 19
        READ_EFFECTIVE_FTW = 16
        MATCHED_LATENCY_ENABLE = 7

        for dds, cfg in self._cfg_map:
            dds.set_cfr1(ram_enable=cfg.ram_enable,
                         ram_destination=cfg.ram_destination,
                         drg_load_lrr=cfg.drg_enable,
                         drg_autoclear=cfg.drg_enable,
                         osk_enable=cfg.osk_enable)
            # Unfortunately ARTIQ does not expose everything on CFR2
            dds.write32(ad9910._AD9910_REG_CFR2,
                        (1 << ENABLE_SINGLE_TONE_ASF)
                        | (cfg.drg_destination << DRG_DEST)
                        | (cfg.drg_enable << DRG_ENABLE)
                        | (1 << READ_EFFECTIVE_FTW)
                        | (1 << MATCHED_LATENCY_ENABLE))

    @kernel
    def commit_enable(self):
        """Commit the previous enable() call.

        The RAM profile will always start from the designated first address.
        The DRG will start from the start parameter as well.
        """
        # Start profile 0 using the same profile switch update.
        # This is achieved by invoking SPI transactions in the same timestamp.
        now = now_mu()
        for cpld in self._cplds:
            at_mu(now)
            cpld.set_profile(0)
        # The timeline advancement of SPI is **naturally restored** here.
        # Note the ordering of at_mu() and set_profile().

    @kernel
    def disable(self):
        """Set register appropriately for disabling profile 0.

        After the function is called. Both RAM mode and the DRG are STILL active. Commit profile
        disable by calling commit_disable() after.
        """
        for dds, _ in self._cfg_map:
            # Disable RAM, DRG and OSK (if enabled).
            # Single-tone profiles use the profile registers for amplitude control. The state of
            # the OSK does not impact the logic here.
            dds.set_cfr1(ram_enable=0, osk_enable=0)
            dds.set_cfr2(drg_enable=0)

    @kernel
    def commit_disable(self):
        """Commit the previous disable() call.

        After this call has taken effect (in terms of the ARTIQ timeline), single-tone profiles on
        profile 7 are enabled instead.
        """
        # End profile 0 using the same profile switch update.
        # This is achieved by invoking SPI transactions in the same timestamp.
        now = now_mu()
        for cpld in self._cplds:
            at_mu(now)
            cpld.set_profile(7)
        # The timeline advancement of SPI is **naturally restored** here.
        # Note the ordering of at_mu() and set_profile().
