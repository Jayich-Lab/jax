import pickle
import time

from artiq.applets.simple import SimpleApplet
from jax import JaxApplet
from jax.tools.applets.dds_channel import DDSChannel, DDSParameters
from jax.util.ui.custom_list_widget import CustomListWidget
from PyQt5 import QtCore, QtGui, QtWidgets


class DDS(QtWidgets.QDockWidget, JaxApplet):
    # signal emitted after getting DDS parameters.
    # a signal is needed to run self.initialize_channels on the default thread.
    # widgets can only be created in the default thread.
    do_initialize = QtCore.pyqtSignal()

    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self.setObjectName("DDS")
        self.setDisabled(
            True
        )  # start with the applet disabled, until artiq server is connected.
        self.do_initialize.connect(self.initialize_channels)

        self.initialize_gui()
        self.load_config_file("dds", args)
        # connects to LabRAD in a different thread, and calls self.labrad_connected when finished.
        self.connect_to_labrad(args.ip)

    def initialize_gui(self):
        self.list_widget = CustomListWidget()
        self.setWidget(self.list_widget)

    async def labrad_connected(self):
        """Called when LabRAD is connected."""
        await self.artiq_connected()
        await self.setup_cxn_listeners()

    async def artiq_connected(self):
        self.artiq = self.cxn.get_server("artiq")
        initialize_now = await self.artiq.is_dds_initialized()
        if initialize_now:
            await self.get_dds_parameters()

        SIGNALID = 124890
        await self.artiq.on_dds_change(SIGNALID)
        self.artiq.addListener(listener=self._dds_changed, source=None, ID=SIGNALID)
        await self.artiq.on_dds_initialize(SIGNALID + 1)
        self.artiq.addListener(
            listener=self._dds_initialized, source=None, ID=SIGNALID + 1
        )

    async def get_dds_parameters(self):
        self.params = await self.artiq.get_dds_parameters()
        self.params = pickle.loads(self.params)
        # tells the main thread that it can populate the DDS channels.
        self.do_initialize.emit()
        self.setDisabled(False)

    @QtCore.pyqtSlot()
    def initialize_channels(self):
        self.channels = {}
        self.list_widget.clear()
        for channel in self.params:
            cpld = "Not implemented"  # current code does not query the cpld name.
            frequency = self.params[channel][0]
            phase = self.params[channel][1]
            amp = self.params[channel][2]
            att = self.params[channel][3]
            state = self.params[channel][4] > 0
            channel_param = DDSParameters(
                self, channel, cpld, amp, att, frequency, phase, state
            )
            channel_widget = DDSChannel(channel_param, self)
            self.channels[channel] = channel_widget
            self._still_looping = False

            self.list_widget.add_item_and_widget(channel, channel_widget)

        if "list_widget" not in self.config:
            self.config["list_widget"] = {}
        self.list_widget_reordered(
            self.list_widget.set_visibility_and_order(self.config["list_widget"])
        )
        self.list_widget.visibility_and_order_changed.connect(
            self.list_widget_reordered
        )

    def list_widget_reordered(self, widget_config):
        self.config["list_widget"] = widget_config
        self.save_config_file()

    async def setup_cxn_listeners(self):
        self.cxn.add_on_connect("artiq", self.run_in_labrad_loop(self.artiq_connected))
        self.cxn.add_on_disconnect("artiq", self.artiq_disconnected)

    def artiq_disconnected(self):
        self.setDisabled(True)

    def _dds_changed(self, signal, value):
        channel, attribute, val = value
        if attribute == "frequency":
            self.channels[channel].on_monitor_freq_changed(val)
        elif attribute == "amplitude":
            self.channels[channel].on_monitor_amp_changed(val)
        elif attribute == "attenuation":
            self.channels[channel].on_monitor_att_changed(val)
        elif attribute == "state":
            self.channels[channel].on_monitor_switch_changed(val > 0.0)

    def _dds_initialized(self, signal, value):
        self.run_in_labrad_loop(self.get_dds_parameters)()


def main():
    applet = SimpleApplet(DDS)
    DDS.add_labrad_ip_argument(applet)  # adds IP address as an argument.
    DDS.add_id_argument(applet)
    applet.run()


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    main()
