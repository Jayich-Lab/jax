from artiq.applets.simple import SimpleApplet
from jax import JaxApplet
from PyQt5 import QtCore, QtGui, QtWidgets


class PMT(QtWidgets.QWidget, JaxApplet):
    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self._dv_on = False
        self._pmt_on = False
        self.set_disable_state()

        self._normal_mode_text = "Normal"
        self._differential_mode_text = "Differential"
        self._pmt_counts_dataset = "pmt.counts_kHz"

        self.initialize_gui()
        # connects to LabRAD in a different thread, and calls self.labrad_connected when finished.
        self.connect_to_labrad(args.ip)

    def set_disable_state(self):
        if self._pmt_on and self._dv_on:
            self.setDisabled(False)
        else:
            self.setDisabled(True)

    def initialize_gui(self):
        shell_font = "MS Shell Dlg 2"

        layout = QtWidgets.QGridLayout()
        self.number = QtWidgets.QLCDNumber()
        self.number.setDigitCount(4)
        self.number.setSmallDecimalPoint(True)
        layout.addWidget(self.number, 0, 0)

        mode_label = QtWidgets.QLabel("Mode:")
        mode_label.setAlignment(QtCore.Qt.AlignBottom)
        mode_label.setFont(QtGui.QFont(shell_font, pointSize=12))
        mode_label.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        layout.addWidget(mode_label, 1, 0)

        self.mode_combobox = QtWidgets.QComboBox()
        self.mode_combobox.addItem(self._normal_mode_text)
        self.mode_combobox.addItem(self._differential_mode_text)
        self.mode_combobox.setFont(QtGui.QFont(shell_font, pointSize=12))
        self.mode_combobox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        layout.addWidget(self.mode_combobox, 2, 0)

        interval_label = QtWidgets.QLabel("Interval:")
        interval_label.setAlignment(QtCore.Qt.AlignBottom)
        interval_label.setFont(QtGui.QFont(shell_font, pointSize=12))
        interval_label.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        layout.addWidget(interval_label, 3, 0)

        self.interval_spinbox = QtWidgets.QDoubleSpinBox()
        self.interval_spinbox.setSuffix(" s")
        self.interval_spinbox.setSingleStep(0.1)
        self.interval_spinbox.setDecimals(2)
        self.interval_spinbox.setFont(QtGui.QFont(shell_font, pointSize=12))
        self.interval_spinbox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        layout.addWidget(self.interval_spinbox, 4, 0)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setFont(QtGui.QFont(shell_font, pointSize=12))
        self.start_button.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        self.start_button.setCheckable(True)
        layout.addWidget(self.start_button, 5, 0)
        self.setLayout(layout)

    async def labrad_connected(self):
        self.setup_gui_listeners()
        await self.vault_connected()
        await self.pmt_connected()
        await self.setup_cxn_listeners()

    async def vault_connected(self):
        self.dv = self.cxn.get_server("vault")
        self._dv_on = True
        await self.dv.subscribe_to_shared_dataset(self._pmt_counts_dataset)
        SHARED_DATA_CHANGE = 128936
        await self.dv.on_shared_data_change(SHARED_DATA_CHANGE)
        self.dv.addListener(
            listener=self._data_change, source=None, ID=SHARED_DATA_CHANGE
        )
        self.set_disable_state()

    async def pmt_connected(self):
        self.pmt = self.cxn.get_server("pmt")
        self._pmt_on = True
        interval_range = await self.pmt.get_interval_range()
        self.interval_spinbox.setRange(*interval_range)
        interval = await self.pmt.get_interval()
        self._set_pmt_interval(interval)
        differential_mode = await self.pmt.is_differential_mode()
        self._set_pmt_mode(differential_mode)
        is_running = await self.pmt.is_running()
        self._set_pmt_state(is_running)

        NEW_MODE = 128937
        await self.pmt.on_new_mode(NEW_MODE)
        self.pmt.addListener(listener=self._new_pmt_mode, source=None, ID=NEW_MODE)
        NEW_INTERVAL = 128938
        await self.pmt.on_new_interval(NEW_INTERVAL)
        self.pmt.addListener(
            listener=self._new_pmt_interval, source=None, ID=NEW_INTERVAL
        )
        FILE_HALF_FULL = 128939
        await self.pmt.on_file_half_full(FILE_HALF_FULL)
        self.pmt.addListener(
            listener=self._file_half_full, source=None, ID=FILE_HALF_FULL
        )
        AUTO_NEW_FILE = 128940
        await self.pmt.on_auto_new_file(AUTO_NEW_FILE)
        self.pmt.addListener(
            listener=self._auto_new_file, source=None, ID=AUTO_NEW_FILE
        )
        START_STOP = 128941
        await self.pmt.on_start_and_stop(START_STOP)
        self.pmt.addListener(
            listener=self._on_start_and_stop, source=None, ID=START_STOP
        )
        self.set_disable_state()

    async def setup_cxn_listeners(self):
        self.cxn.add_on_connect("pmt", self.run_in_labrad_loop(self.pmt_connected))
        self.cxn.add_on_disconnect("pmt", self.pmt_disconnected)

        self.cxn.add_on_connect("vault", self.run_in_labrad_loop(self.vault_connected))
        self.cxn.add_on_disconnect("vault", self.vault_disconnected)

    def setup_gui_listeners(self):
        self.number.overflow.connect(self.number_overflow)
        self.start_button.toggled.connect(self.start_button_toggled)
        self.mode_combobox.currentTextChanged.connect(self.mode_combobox_text_changed)
        self.interval_spinbox.valueChanged.connect(self.interval_spinbox_value_changed)

    def pmt_disconnected(self):
        self._pmt_on = False
        self.set_disable_state()

    def vault_disconnected(self):
        self._dv_on = False
        self.set_disable_state()

    def _data_change(self, signal, value):
        if value[1] == self._pmt_counts_dataset:
            self._set_number(value[2][0][0])

    def _new_pmt_mode(self, signal, value):
        self._set_pmt_mode(value)

    def _new_pmt_interval(self, signal, value):
        self._set_pmt_interval(value)

    def _file_half_full(self, signal, value):
        print("PMT file half full")

    def _auto_new_file(self, signal, value):
        print("New PMT file automatically created.")

    def _on_start_and_stop(self, signal, value):
        self._set_pmt_state(value)

    def _set_number(self, counts):
        self.number.display(counts)

    def _set_pmt_interval(self, interval):
        self.interval_spinbox.blockSignals(True)
        self.interval_spinbox.setValue(interval)
        self.interval_spinbox.blockSignals(False)

    def _set_pmt_mode(self, is_differential_mode):
        self.mode_combobox.blockSignals(True)
        if is_differential_mode:
            self.mode_combobox.setCurrentText(self._differential_mode_text)
        else:
            self.mode_combobox.setCurrentText(self._normal_mode_text)
        self.mode_combobox.blockSignals(False)

    def _set_pmt_state(self, is_running):
        self.start_button.blockSignals(True)
        self.start_button.setChecked(is_running)
        if is_running:
            self.start_button.setText("Stop")
        else:
            self.start_button.setText("Start")
        self.start_button.blockSignals(False)

    def number_overflow(self):
        self.number.display("OUFL")

    def start_button_toggled(self, checked):
        async def _start_button_toggled(self, checked):
            self._set_pmt_state(checked)
            if checked:
                await self.pmt.start()
            else:
                await self.pmt.stop()

        self.run_in_labrad_loop(_start_button_toggled)(self, checked)

    def mode_combobox_text_changed(self, text):
        async def _mode_combobox_text_changed(self, text):
            await self.pmt.set_mode(text == self._differential_mode_text)

        self.run_in_labrad_loop(_mode_combobox_text_changed)(self, text)

    def interval_spinbox_value_changed(self, value):
        async def _interval_spinbox_value_changed(self, value):
            await self.pmt.set_interval(value)

        self.run_in_labrad_loop(_interval_spinbox_value_changed)(self, value)


def main():
    applet = SimpleApplet(PMT)
    PMT.add_labrad_ip_argument(applet)  # adds IP address as an argument.
    applet.run()


if __name__ == "__main__":
    main()
