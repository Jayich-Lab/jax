import pickle
from artiq.experiment import *
from jax import JaxExperiment, SinaraEnvironment


__all__ = ["InitializeSinara"]


class InitializeSinara(JaxExperiment, SinaraEnvironment):
    """Handles device initialization.

    This sets previously initialized devices to their original parameter settings.
    For new uninitialized devices it saves their parameters.

    To use this class, inherit this class in the experiment repository.

    A device is a DDS channel or a TTL channel.
    Parameters can be frequency, phase, amplitude, state, etc.

    This experiment assumes that the sinara hardware has at least a DDS and a TTL board.
    For an uninitialized DDS or TTL device, the experiment assumes its state to be off (low).
    """
    kernel_invariants = {
        "initialized_ddses", "uninitialized_ddses", "initialized_ttls", "uninitialized_ttls"
    }

    def build(self):
        super().build()
        second_lowest_priority = -99  # we use priorities range from -100 to 100.
        self.set_default_scheduling(priority=second_lowest_priority, pipeline_name="main")

    def prepare(self):
        super().prepare()
        self.get_dds_and_ttls()

    def run(self):
        self.run_kernel()
        self.cxn.artiq.finish_dds_initialize()
        self.cxn.artiq.finish_ttl_initialize()
        self.disconnect_labrad()

    @host_only
    def get_dds_and_ttls(self):
        """Gets DDS and TTL objects that are initialized and uninitialized.

        Populates self.initialized_ddses, self.uninitialized_ddses,
        self.initialized_ttls, and self.uninitialized_ttls. The first element
        in each of these lists is a placeholder element for determining element type.
        """
        self.urukuls = [self.get_device(kk) for kk in self.devices.urukuls]
        dds_params = pickle.loads(self.cxn.artiq.get_dds_parameters())
        dummy_dds = self.get_device(self.devices.ad9910s[0])
        self.initialized_ddses = [(dummy_dds, [0., 0., 0., 0., 0.])]  # DDSes in artiq server.
        self.uninitialized_ddses = [(dummy_dds, "")]  # DDSes not in artiq server.
        for name in dds_params:
            if name in self.devices.ad9910s:
                self.initialized_ddses.append((self.get_device(name), dds_params[name]))
            else:
                self.cxn.artiq.remove_sinara_dds(name)
        for name in self.devices.ad9910s:
            if name not in dds_params:
                self.uninitialized_ddses.append((self.get_device(name), name))

        ttl_params = pickle.loads(self.cxn.artiq.get_ttl_parameters())
        dummy_ttl = self.get_device(self.devices.ttl_outs[0])
        self.initialized_ttls = [(dummy_ttl, 0.)]
        self.uninitialized_ttls = [(dummy_ttl, "")]
        for name in ttl_params:
            if name in self.devices.ttl_outs:
                self.initialized_ttls.append((self.get_device(name), ttl_params[name]))
            else:
                self.cxn.artiq.remove_sinara_ttl(name)
        for name in self.devices.ttl_outs:
            if name not in ttl_params:
                self.uninitialized_ttls.append((self.get_device(name), name))

    @kernel
    def run_kernel(self):
        self.core.reset()
        for kk in range(len(self.urukuls)):
            self.core.break_realtime()
            self.urukuls[kk].init()

        for kk in range(len(self.initialized_ddses)):
            if kk > 0:
                dds, values = self.initialized_ddses[kk]
                self.core.break_realtime()
                dds.init()
                self._set_dds(dds, values)

        for kk in range(len(self.uninitialized_ddses)):
            if kk > 0:
                dds, name = self.uninitialized_ddses[kk]
                self.core.break_realtime()
                dds.init()
                self.get_dds(dds, name)

        for kk in range(len(self.initialized_ttls)):
            if kk > 0:
                ttl, value = self.initialized_ttls[kk]
                self._set_ttl(ttl, value)

        for kk in range(len(self.uninitialized_ttls)):
            if kk > 0:
                ttl, name = self.uninitialized_ttls[kk]
                self.get_ttl(ttl, name)

    @kernel
    def get_dds(self, device, name):
        """Get the DDS values from the sinara hardware.

        It gets the frequency, phase, amplitude, and attenuation from the DDS,
        and assumes that the DDS state is -1. (off).
        """
        self.core.break_realtime()
        frequency, phase, amplitude = device.get()
        self.core.break_realtime()
        attenuation = device.get_att()
        state = -1.
        self.update_dds(name, [frequency, phase, amplitude, attenuation, state])

    @rpc
    def update_dds(self, name: TStr, values: TList(TFloat)):
        self.cxn.artiq.update_sinara_dds_value_from_init_experiment(name, values)

    @kernel
    def get_ttl(self, device, name):
        """Assumes that the TTL state is -1. (low)."""
        self.update_ttl(name, -1.)

    @rpc
    def update_ttl(self, name: TStr, value: TFloat):
        self.cxn.artiq.update_sinara_ttl_value_from_init_experiment(name, value)
