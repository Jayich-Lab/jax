import pickle as _p
import time as _t

from artiq.experiment import *
from jax import InfiniteLoop, SinaraEnvironment

__all__ = ["IDLE"]


class IDLE(InfiniteLoop, SinaraEnvironment):
    """Base class for a background running experiment that reads PMT and sets TTL/DDS parameters.

    Inherit this class in the experiment repository and define the class variables:
        REPUMP_AOM_CHANNELS: list of strs, names of DDSes controlling repump lasers.
            To control ions that have hyperfine structures, multiple repump DDSes may be necessary.
        PMT_EDGECOUNTER: str, edgecounter device for PMT input.

    This experiment assumes that the device has at least one AD9910 DDS and at least one TTL board.
    """
    DIFFERENTIAL_TRIGGER = None
    REPUMP_AOM_CHANNELS = None
    kernel_invariants = {
        "DIFFERENTIAL_TRIGGER", "REPUMP_AOM_CHANNELS", "PMT_EDGECOUNTER", "repump_aoms",
        "pmt_counter", "ad9910s", "ttl_outs"
    }

    def build(self):
        super().build()
        lowest_priority = -100  # we use priorities range from -100 to 100.
        self.set_default_scheduling(priority=lowest_priority, pipeline_name="main")

        self.rising_pulse = True
        if self.DIFFERENTIAL_TRIGGER is None:
            raise Exception("DIFFERENTIAL_TRIGGER must be defined.")
        self.differential_trigger = self.get_device(self.DIFFERENTIAL_TRIGGER)
        if self.REPUMP_AOM_CHANNELS is None:
            raise Exception("REPUMP_AOM_CHANNELS must be defined.")
        self.repump_aoms = [self.get_device(kk) for kk in self.REPUMP_AOM_CHANNELS]
        self._get_all_dds_and_ttl_objects()

    def prepare(self):
        super().prepare()
        self._get_repump_aom_states()

    def host_startup(self):
        pass

    def host_cleanup(self):
        self.disconnect_labrad()

    @host_only
    def _get_all_dds_and_ttl_objects(self):
        self.ad9910s = []
        for kk in self.devices.ad9910s:
            self.ad9910s.append(self.get_device(kk))
        self.ttl_outs = []
        for kk in self.devices.ttl_outs:
            self.ttl_outs.append(self.get_device(kk))

    @host_only
    def _get_repump_aom_states(self):
        """Gets the states of repump AOMs.

        If a repump AOM is set to off, don't turn it on during the differential mode sequence.
        """
        dds_params = _p.loads(self.cxn.artiq.get_dds_parameters())
        self.repump_aom_states = []
        for kk in self.REPUMP_AOM_CHANNELS:
            self.repump_aom_states.append(dds_params[kk][-1])

    @kernel
    def kernel_before_loops(self):
        self.core.reset()
        self.core.break_realtime()

    @kernel
    def kernel_loop(self, loop_index: TInt64):
        self.update_hardware()
        self.core.break_realtime()

        max_wait_time_mu = self.get_max_cycle()
        total_trigger_wait_time_mu = 0
        trigger_time = -1

        # we will go through the whole time in 50 us intervals
        trigger_cycle_mu = self.exp.core.seconds_to_mu(50 * us)

        self.differential_trigger.count(now_mu())  # clears all existing timestamps.

        if self.rising_pulse:
            self._gate_func = self.differential_trigger.gate_rising_mu
        else:
            self._gate_func = self._trigger_ttl.gate_falling_mu

        # waits for a trigger for trigger_cycle_mu.
        self._gate_func(trigger_cycle_mu)
        total_trigger_wait_time_mu += trigger_cycle_mu

        while (total_trigger_wait_time_mu < max_wait_time_mu - trigger_cycle_mu):
            # wait for a trigger in the cycle
            gate_end_time_mu = self._gate_func(trigger_cycle_mu)
            total_trigger_wait_time_mu += trigger_cycle_mu

            # check for a trigger
            trigger_time = self.differential_trigger.timestamp_mu(
                gate_end_time_mu - trigger_cycle_mu
                )
            if trigger_time > 0:
                break

        # takes care of the last section of time
        if trigger_time < 0:
            gate_end_time_mu = self._gate_func(max_wait_time_mu - total_trigger_wait_time_mu)
            trigger_time = self.differential_trigger.timestamp_mu(gate_end_time_mu)

        if trigger_time > 0:  # if trigger is detected at some point
            self.switch_repump_aom_states(self.rising_pulse)

            # switch to the next pulse
            self.rising_pulse = not self.rising_pulse

    @kernel(flags={"fast-math"})
    def update_hardware(self):
        """Checks whether DDS/TTL needs to be updated and update them."""
        dds_changes = self.get_dds_changes()
        self.core.break_realtime()
        for kk in range(len(dds_changes)):
            if kk > 0:
                index, index_repump, name, values = dds_changes[kk]
                self.update_dds(index, index_repump, name, values)

        ttl_changes = self.get_ttl_changes()
        self.core.break_realtime()
        for kk in range(len(ttl_changes)):
            if kk > 0:
                index, value = ttl_changes[kk]
                self.update_ttl(index, value)
        self.core.break_realtime()

    @rpc
    def get_dds_changes(self) -> TList(TTuple([TInt32, TInt32, TStr, TFloat])):
        """Gets all pending changes for DDSes.

        If there is no change, it must have a placeholder element in the list to ensure that
        ARTIQ python can recognize its type.

        Returns:
            DDS changes. The first element is always a placeholder.
                The second int in the tuple is index in self.repump_aoms. This is used to
                determine whether repump AOMs should be turned on in the differential mode.
        """
        dds_changes = self.cxn.artiq.get_dds_change_queues()
        to_kernel = [(-1, -1, "placeholder", 0.)]
        for kk in dds_changes:
            index = self.devices.ad9910s.index(kk[0])
            try:
                index_repump = self.REPUMP_AOM_CHANNELS.index(kk[0])
            except ValueError as e:
                index_repump = -1
            to_kernel.append((index, index_repump, kk[1], kk[2]))
        return to_kernel

    @rpc
    def get_ttl_changes(self) -> TList(TTuple([TInt32, TFloat])):
        """Gets all pending changes for TTLs.

        If there is no change, it must have a placeholder element in the list to ensure that
        ARTIQ python can recognize its type.

        Returns:
            TTL changes. The first element is always a placeholder.
        """
        ttl_changes = self.cxn.artiq.get_ttl_change_queues()
        to_kernel = [(-1, 0.)]
        for kk in ttl_changes:
            index = self.devices.ttl_outs.index(kk[0])
            to_kernel.append((index, kk[1]))
        return to_kernel

    @kernel(flags={"fast-math"})
    def update_dds(self, index: TInt32, index_repump: TInt32, attribute: TStr, value: TFloat):
        """Sets DDS value."""
        device = self.ad9910s[index]
        if attribute == "frequency":
            freq, phase, amp = device.get()
            self.core.break_realtime()
            delay_mu(self.dds_set_delay_mu)
            device.set(value, phase, amp)
            self.core.break_realtime()
        elif attribute == "phase":
            freq, phase, amp = device.get()
            self.core.break_realtime()
            delay_mu(self.dds_set_delay_mu)
            device.set(freq, value, amp)
            self.core.break_realtime()
        elif attribute == "amplitude":
            freq, phase, amp = device.get()
            self.core.break_realtime()
            delay_mu(self.dds_set_delay_mu)
            device.set(freq, phase, value)
            self.core.break_realtime()
        elif attribute == "attenuation":
            # get_att_mu() required to correctly set the atts of other DDSes of the same urukul.
            device.get_att_mu()
            self.core.break_realtime()
            device.set_att(value)
            self.core.break_realtime()
        elif attribute == "state":
            is_on = value > 0.
            if is_on:
                device.sw.on()
            else:
                device.sw.off()
            if index_repump >= 0:
                self.repump_aom_states[index_repump] = value
            self.core.break_realtime()

    @kernel(flags={"fast-math"})
    def update_ttl(self, index: TInt32, value: TFloat):
        """Sets TTL value."""
        device = self.ttl_outs[index]
        if value > 0.:
            device.on()
        else:
            device.off()
        self.core.break_realtime()

    @rpc
    def switch_repump_aom_states(self, turn_off):
        for kk in range(len(self.repump_aoms)):
            # if the repump AOM is off, don't turn on and off the AOM.
            # the repump AOM stays off for both differential high and low counting periods.

            if self.repump_aom_states[kk] > 0.:
                if turn_off:
                    self.repump_aoms[kk].sw.off()
                else:
                    self.repump_aoms[kk].sw.on()

    @rpc
    def get_max_cycle(self):
        """Gets PMT counting interval.

        This is the max wait time for differential mode. If PMT is not on,
        this defaults to 0."""
        try:
            interval = self.cxn.pmt_arduino.get_interval()
            self.interval_ms = interval / ms
            interval_mu = self.core.seconds_to_mu(interval)
            if not self.cxn.pmt_arduino.is_running():
                interval_mu = 0
            return interval_mu
        except Exception as e:
            pass
        return 0
