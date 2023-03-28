from enum import Enum

import numpy as np
from artiq.coredevice import ad9910
from artiq.experiment import kernel


class DRGType(Enum):
    """The type of digital ramp.

    Enums:
        FREQ: Data are treated as frequencies (Hz). The DRG will generate a frequency ramp.
        PHASE: Data are treated as phases (turns). The DRG will generate a phase ramp.
        AMP: Data are treated as amplitudes. The DRG will generate an amplitude ramp.
    """
    FREQ = 0
    PHASE = 1
    AMP = 2


class DRG:
    """Digital Ramp Generator (DRG).

    The Digital Ramp Generator (DRG) is a module that generates a positive ramp of a DDS parameter
    (frequency/amplitude/phase). First, the DRG initializes the parameter at a user-defined
    starting point. At an user-defined time interval, the DRG increments the DDS parameter until it
    reaches the user-specified limit (elaborated below).

    However, since ARTIQ does not support DRG operations, and the CPLD logic fixes DRG to decrement
    the DDS parameters, we use the workaround currently in the ARTIQ codebase.

    Link to workaround code: See #72.

    The workaround allows the generation of positive ramp ONLY by overflowing registers. After
    reaching the upper limit (`end`), the DRG stays at a fixed level until the DRG is reset (see
    `dwell_high` explanations below for more details). Hence, only unidirectional positive ramp is
    supported.

    Note: Enabling DRG bidirectional sweep will require both Urukul CPLD and ARTIQ gateware
    developments, such that the direction pin (DRCTL) can be controlled by ARTIQ. It should either
    be indirect control (through CPLD register, high timing overhead) or direct control (ARTIQ TTL
    modules, like RF switches).

    The workaround leads to an edge case behavior when the DRG accumulator exceeds the upper limit.
    If the accumulator also reaches/exceeds the lower limit (by overflowing), the DRG accumulator
    will reset to lower limit (dwell_high=False) instead of staying at the upper limit (dwell_high=
    True).

    In terms of the DRG API, the waveform stalls at `end` if `dwell_high` is True, and `start`-
    "step gap" if `dwell_high` is False.

    Note: "step gap" refers to the difference between the value of each step. For example, consider
    an amplitude DRG. start=0.1, end=0.9, num_of_steps=9. "step gap" in high level sense would be
    0.1. Note that this class performs such calculation in machine units instead of user arguments
    for the sake of correctness.

    Attributes:
        dest: int, the DRG data destination.
        rate: int, the number of sysclk cycles per RAM step.
        high: int32, DRG upper limit field. Specifies the upper limit of the digital ramp.
        step: int32, DRG step increment. Specifies the increment of the parameter in machine unit.
        low: int32, DRG lower limit field. Specifies the lower limit of the digital ramp.
        drg_type: DRGType, the type of DRG.

    Args:
        dds: AD9910, the DDS that will playback the digital ramp.
        start: float, the starting value of the DDS parameter (frequency/amplitude/phase). It must
            be at least a "step gap" above 0.
        end: float, the final value of the DDS parameter (frequency/amplitude/phase).
            If dwell_high=True, DRG will generate `start` - "step gap" after this value.
        ramp_interval: float, the time interval between each step of the DRG playback. Keep the
            interval at a multiple of 4*T_sysclk (4*1 ns).
        drg_type: DRGType, the type of DRG.
        num_of_steps: int, the number of steps in the DRG if specified. Exactly ONE of `num_of_step`
            and `ramp_step` should be specified. Defaults to None.
        step_gap: float or string, the specification of the difference between the value of each
            step of the DRG. It takes one of the following 3 forms:
            1. The specific step gap (i.e. the increment of value per ramp interval) as float.
            2. The string "fine" to specify smallest step gap for the specific parameter.
            3. None (default). The ramp step is not specified.
            Exactly ONE of `num_of_step` and `step_gap` should be specified. Defaults to None.
        dwell_high: bool, the DDS parameter will stall at `end` if True. This parameter is mainly
            to check the user's understanding of the dwell high condition. Defaults to True.

    Raises:
        ValueError: Invalid DRG parameters.
    """
    def __init__(self, dds, start, end, ramp_interval, drg_type,
                 num_of_steps=None, step_gap=None, dwell_high=True):
        if (num_of_steps is not None) == (step_gap is not None):
            raise ValueError("num_of_step and step_gap are both specified or unspecified")
        self.dest = drg_type.value
        self.rate = int(ramp_interval * dds.sysclk / 4.0)

        # DRG can only start from `low` + `step`.
        # Store everything as uint32 for easier math and verifications.
        if drg_type == DRGType.FREQ:
            start_mu = dds.frequency_to_ftw(start).astype('uint32')
            self.high = dds.frequency_to_ftw(end).astype('uint32')

            if step_gap is not None:
                if step_gap == "fine":
                    # Both DRG accuumlator and FTW have 32-bits resolution.
                    # DDS core fully reflects the DRG resolution.
                    self.step = np.uint32(1)
                else:
                    self.step = dds.frequency_to_ftw(step_gap).astype('uint32')

        elif drg_type == DRGType.PHASE:
            # DRG registers are 32-bits. POW register is 16-bits.
            # Using turns_to_pow will result in the loss of precision in DRG step.
            def _turns_to_drg_args(turns):
                pow_ = round(turns * (2**32))
                if pow_ < 0 or pow_ > (2**32 - 1):
                    raise ValueError("Phase parameter does not wrap around in DRG")
                return np.uint32(pow_)

            start_mu = _turns_to_drg_args(start)
            self.high = _turns_to_drg_args(end)

            if step_gap is not None:
                if step_gap == "fine":
                    # DRG has 32-bits resolution, but POW only has 16.
                    # Only the most-significant 16-bits can reach the DDS core.
                    self.step = np.uint32(1 << 16)
                else:
                    self.step = _turns_to_drg_args(step_gap).astype('uint32')

        elif drg_type == DRGType.AMP:
            # DRG registers are 32-bits. ASF field in the ASF register is 14-bits.
            # Using amplitude_to_asf will result in the loss of precision in DRG step.
            def _amplitude_to_drg_regs(amplitude):
                # Note the difference between amplitude and phase conversion.
                # Amplitude scale factor does not overflow at 1.0; Phase offset does.
                if amplitude < 0.0 or amplitude > 1.0:
                    raise ValueError("Amplitude parameter exceeds DRG limits")
                return np.uint32(amplitude * 0xFFFFFFFF)

            start_mu = _amplitude_to_drg_regs(start)
            self.high = _amplitude_to_drg_regs(end)

            if step_gap is not None:
                if step_gap == "fine":
                    # DRG has 32-bits resolution, but ASF only has 14.
                    # Only the most-significant 18-bits can reach the DDS core.
                    self.step = np.uint32(1 << 18)
                else:
                    self.step = _amplitude_to_drg_regs(step_gap).astype('uint32')

        else:
            raise ValueError("Invalid DRG type argument")

        if self.high <= start_mu:
            raise ValueError("Upper limit of DRG must be higher than the lower limit of DRG")

        if num_of_steps is not None:
            self.step = (self.high - start_mu)//(num_of_steps - 1)

        # Verify dwell high condition.
        if dwell_high != ((self.high + self.step) < 2**32):
            raise ValueError("Inconsistent DRG dwell high behavior\n"
                             + "Hint: Try dwell_high={}".format(not dwell_high))
        # Verify start param validity
        if start_mu < self.step:
            raise ValueError("start value too low")

        self.low = start_mu - self.step

        # Cast DRG parameters from uint32 to int32. ARTIQ API only accepts int32.
        for arg in ["low", "high", "step"]:
            setattr(self, arg, getattr(self, arg).astype('int32'))

        self.drg_type = drg_type


class DRGMap:
    """A mapping between DRG configuration and the DDS that performs the playback.
    """
    def __init__(self):
        self._drg_map = []

    def append(self, dds, drg):
        """Append a RAM profile into the builder.

        Args:
            dds: AD9910, the DDS that will playback the digital ramp.
            drg: DRG, a Digital Ramp Generator (DRG) configuration.
        """
        self._drg_map.append((dds, drg))

    @kernel
    def load(self):
        """Loads the mapped DRG configuration to the corresponding DDS.

        The configuration is inactive until a change of profile/ IO update pulse.
        This is handled by the AD9910Manager commit methods.
        """
        for dds, drg in self._drg_map:
            dds.write64(ad9910._AD9910_REG_RAMP_LIMIT, drg.high, drg.low)
            dds.write64(ad9910._AD9910_REG_RAMP_STEP, -drg.step, 0)
            dds.write32(ad9910._AD9910_REG_RAMP_RATE, drg.rate << 16)
