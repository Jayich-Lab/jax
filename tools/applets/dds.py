import time
from sipyco import pyon
from PyQt5 import QtGui, QtWidgets, QtCore
from artiq.applets.simple import SimpleApplet
from jax import JaxApplet
from jax.tools.applets.dds_channel import DDSChannel, DDSParameters


class DDS(QtWidgets.QWidget, JaxApplet):
    # signal emitted after getting DDS parameters.
    # a signal is needed to run self.initialize_channels on the default thread.
    # widgets can only be created in the default thread.
    do_initialize = QtCore.pyqtSignal()

    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self.setDisabled(True)  # start with the applet disabled, until artiq server is connected.
        self.do_initialize.connect(self.initialize_channels)

        self.initialize_gui()
        # connects to LabRAD in a different thread, and calls self.labrad_connected when finished.
        self.connect_to_labrad(args.ip)

    def initialize_gui(self):
        font = QtGui.QFont("Arial", 15)
        layout = QtWidgets.QGridLayout()
        self.grid = QtWidgets.QListWidget()
        self.grid.setAcceptDrops(False)
        self.grid.setDragEnabled(False)
        self.grid.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.grid.setFlow(QtWidgets.QListView.LeftToRight)
        self.grid.setResizeMode(QtWidgets.QListView.Adjust)
        self.grid.setViewMode(QtWidgets.QListView.IconMode)
        layout.addWidget(self.grid)
        self.setLayout(layout)

    async def labrad_connected(self):
        """Called when LabRAD is connected."""
        await self.artiq_connected()
        await self.setup_cxn_listeners()

    async def artiq_connected(self):
        self.artiq = self.cxn.get_server("artiq")
        self.params = await self.artiq.get_dds_parameters()
        self.params = pyon.decode(self.params)
        self.do_initialize.emit()  # tells the main thread that it can populate the DDS channels.

        DDS_CHANGE = 124890
        await self.artiq.on_dds_change(DDS_CHANGE)
        self.artiq.addListener(listener=self._dds_changed, source=None, ID=DDS_CHANGE)
        self.setDisabled(False)

    @QtCore.pyqtSlot()
    def initialize_channels(self):
        self.channels = {}
        for channel in self.params:
            cpld = "Not implemented"  # current code does not query the cpld name.
            frequency = self.params[channel][0]
            phase = self.params[channel][1]
            amp = self.params[channel][2]
            att = self.params[channel][3]
            state = (self.params[channel][4] > 0)
            channel_param = DDSParameters(self, channel, cpld, amp, att, frequency, phase, state)
            channel_widget = DDSChannel(channel_param, self)
            self.channels[channel] = channel_widget
            self._still_looping = False

            item = QtWidgets.QListWidgetItem()
            size = channel_widget.sizeHint()
            padding = 10
            new_size = QtCore.QSize(size.width() + padding, size.height() + padding)
            item.setSizeHint(size)
            self.grid.setGridSize(new_size)
            self.grid.addItem(item)
            self.grid.setItemWidget(item, channel_widget)

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
            self.channels[channel].on_monitor_switch_changed(val > 0.)


def main():
    applet = SimpleApplet(DDS)
    DDS.add_labrad_ip_argument(applet)  # adds IP address as an argument.
    applet.run()


if __name__ == "__main__":
    main()
