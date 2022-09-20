import os as _os
import time as _t

import jax

from pydux.analysis.units import MHz
from pydux.lab_config import ComputerIPs


class WavemeterMonitor(jax.JaxExperiment, jax.JaxEnvironment):
    """Wavemeter Monitoring base experiment

    Args:
        WAVEMETER_PORT: int, wavemeter port that laser connects to
        LASER_NAME: str, wavelength of laser without units, example "461"
        WS8, bool, true is using ws8 wavemeter and false if using moglabs wavemeter
    """

    WAVEMETER_PORT = None
    LASER_NAME = None
    WS8 = True
    ATTRIBUTES = [None, 0]

    def _connect_labrad(self):
        import labrad

        self.cxn = labrad.connect()
        self.cxn_wlm = labrad.connect(
            ComputerIPs.addresses["wellington"], password=_os.environ["LABRADPASSWORD"]
        )
        try:
            self.dv = self.cxn.vault
        except Exception:
            print("Data vault is not connected.")

    def disconnect_labrad(self):
        self.cxn.disconnect()
        self.cxn_wlm.disconnect()

    def build(self):
        super().build()
        self.set_default_scheduling(priority=0, pipeline_name=self.LASER_NAME)

    def prepare(self):
        super().prepare()

    def run(self):
        """Keeps running until the experiment is stopped by the user."""
        self.open_file()
        self._is_dataset_open = True
        if not (self.ATTRIBUTES == [None, 0]):
            self.add_attribute(self.ATTRIBUTES[0], self.ATTRIBUTES[1])
        if self.WS8:
            frequency, self.last_time = self.cxn_wlm.ws8_server.get_frequency(self.WAVEMETER_PORT)
        else:
            frequency, self.last_time = self.cxn_wlm.moglabs_wavemeter.get_frequency(
                self.WAVEMETER_PORT
            )
        self.time_name = self.add_dataset(self.LASER_NAME + ".time", [self.last_time], shared=True)
        self.dataset_name = self.add_dataset(
            self.LASER_NAME + ".frequency_MHz", [frequency / MHz], shared=True
        )
        while True:
            _t.sleep(0.1)
            should_stop = self.check_stop_or_do_pause()
            if should_stop:
                self.close_file()
                self.disconnect_labrad()
                break
            else:
                self.get_wavemeter_frequency()

    def get_wavemeter_frequency(self):
        """
        If the wavemeter has drifted since the last time we checked,
        updates + saves the frequency.
        """
        try:
            if self.WS8:
                frequency, time = self.cxn_wlm.ws8_server.get_frequency(self.WAVEMETER_PORT)
            else:
                frequency, time = self.cxn_wlm.moglabs_wavemeter.get_frequency(self.WAVEMETER_PORT)
        except Exception as e:
            print(e)
        if time != self.last_time:
            self.last_time = time
            self.append_dataset(self.time_name, [self.last_time])
            self.append_dataset(self.dataset_name, [frequency / MHz])
