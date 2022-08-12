import numpy as _np
import pickle as _p
from artiq.experiment import *
from jax import JaxEnvironment
from jax.util.devices import Devices


class SinaraEnvironment(JaxEnvironment):
    """Environment for Jayich lab experiments that use sinara hardware.

    See JaxEnvironment for additional attributes that is not specific to sinara hardware.

    self.devices provides lists of devices and tracks devices used in an experiment.
    A new device manager class can inherit jax.utilities.devices.Devices to support more device
    types, and DEVICE_CLASS must be set to the new device manager class.

    Attributes:
        core: ARTIQ core device.
        devices: Devices, device manager.
        rtio_cycle_mu: np.int64, real-time input/output clock time in machine unit.
            In order for RTIO to distinguish different events, e.g., TTL pulses.
            The time separation between them must be at least this value.
        dds_set_delay_mu: np.int64, slack time needed for an ad9910.set() call.
            This is due to slow float operations in the kasli.
    """
    DEVICE_CLASS = Devices

    def build(self):
        self._get_devices()
        super().build()
        self.setattr_device("core")

    def _get_devices(self):
        """Get a device manager that lists and handles available devices."""
        self.devices = self.DEVICE_CLASS(self.get_device_db())

    def prepare(self):
        super().prepare()
        self.rtio_cycle_mu = _np.int64(self.core.ref_multiplier)
        self.dds_set_delay_mu = self.core.seconds_to_mu(200*us)
        self._preexp_dds_params = _p.loads(self.cxn.artiq.get_dds_parameters())
        self._preexp_ttl_params = _p.loads(self.cxn.artiq.get_ttl_parameters())

    @host_only
    def get_device(self, key):
        """Gets a device and saves it to the device manager.

        This overloads artiq.experiment.HasEnvironment.get_device.
        """
        device = super().get_device(key)
        self.devices.use_device(key, device)
        return device

    @host_only
    def setattr_device(self, key):
        """Sets a device as an attribute of the experiment, and saves it in the device manager.

        self.`key` can be used to access the device.
        This overloads artiq.experiment.HasEnvironment.setattr_device.
        """
        super().setattr_device(key)
        self.devices.use_device(key, getattr(self, key))

    @host_only
    def turn_off_all_ddses(self):
        """Turns off all DDSes used in an experiment."""
        if len(self.devices.ad9910s_used) > 0:
            self._turn_off_ad9910s()

    @host_only
    def reset_sinara_hardware(self):
        """Sets all hardware back to pre-experiment values."""
        if len(self.devices.urukuls_used) > 0:
            self._reset_urukuls()

        if len(self.devices.ad9910s_used) > 0:
            params_used_ad9910s = []
            for kk in self.devices.ad9910s_used:
                params_used_ad9910s.append(self._preexp_dds_params[kk[0]])
            self._reset_ad9910s(params_used_ad9910s)

        if len(self.devices.ttl_outs_used) > 0:
            params_used_ttl_outs = []
            for kk in self.devices.ttl_outs_used:
                params_used_ttl_outs.append(self._preexp_ttl_params[kk[0]])
            self._reset_ttl_outs(params_used_ttl_outs)

    @kernel
    def _turn_off_ad9910s(self):
        """Turns off the rf switches and sets amplitudes to 0."""
        for name, ad9910 in self.devices.ad9910s_used:
            self.core.break_realtime()
            delay_mu(self.rtio_cycle_mu)
            ad9910.sw.off()
            freq, phase, amp = ad9910.get()
            self.core.break_realtime()
            delay_mu(self.dds_set_delay_mu)
            ad9910.set(freq, phase, 0.)

    @kernel
    def _reset_urukuls(self):
        """Sets all urukuls back to profile 7 (the default profile)."""
        for name, urukul in self.devices.urukuls_used:
            self.core.break_realtime()
            delay_mu(self.rtio_cycle_mu)
            urukul.set_profile(7)

    @kernel
    def _reset_ad9910s(self, params):
        """Sets all AD9910s back to pre-experiment parameters."""
        for kk in range(len(self.devices.ad9910s_used)):
            name, ad9910 = self.devices.ad9910s_used[kk]
            values = params[kk]
            self._set_dds(ad9910, values)

    @kernel
    def _reset_ttl_outs(self, params):
        """Sets all TTL outputs back to pre-experiment parameters."""
        for kk in range(len(self.devices.ttl_outs_used)):
            name, ttl = self.devices.ttl_outs_used[kk]
            value = params[kk]
            self._set_ttl(ttl, value)

    @kernel(flags={"fast-math"})
    def _set_dds(self, device, values):
        """Sets frequency, phase, amplitude, attenuation, and state of a DDS.

        Args:
            device: AD9910, DDS device.
            values: list of floats, [frequency, phase, amplitude, attenuation, and state].
        """
        self.core.break_realtime()
        delay_mu(self.rtio_cycle_mu)
        delay_mu(self.dds_set_delay_mu)
        device.set(values[0], values[1], values[2])
        self.core.break_realtime()
        device.get_att_mu()
        self.core.break_realtime()
        delay_mu(self.dds_set_delay_mu)
        device.set_att(values[3])
        self.core.break_realtime()
        if values[4] > 0.:
            device.sw.on()
        else:
            device.sw.off()

    @kernel(flags={"fast-math"})
    def _set_ttl(self, device, value):
        """Sets state of a TTL.

        Args:
            device: TTLOut or TTLInOut, TTL device.
            values: float, state, -1. (off), 1. (on).
        """
        self.core.break_realtime()
        delay_mu(self.rtio_cycle_mu)
        if value > 0.:
            device.on()
        else:
            device.off()

    @kernel(flags={"fast-math"})
    def _set_dac(self, device, value):
        """Sets voltages of a DAC.

        Args:
            device: Zotino or Fastino, DAC device.
            voltages: list of floats, correspond to voltages on pins.
        """
        self.core.break_realtime()
        delay_mu(self.rtio_cycle_mu)
        device.set_dac(value)
