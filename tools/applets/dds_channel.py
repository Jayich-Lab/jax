from jax.util.ui.dialog_on_top import DialogOnTop
from PyQt5 import QtCore, QtGui, QtWidgets


class DDSParameters:
    """Stores DDS parameters and calls the artiq server when changes are made."""

    def __init__(self, parent, channel, cpld, amplitude, att, frequency, phase, state):
        self.parent = parent
        self.channel = channel
        self.cpld = cpld
        self._amplitude = amplitude
        self._att = att
        self._frequency = frequency
        self._phase = phase
        self._state = state

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def att(self):
        return self._att

    @property
    def frequency(self):
        return self._frequency

    @property
    def phase(self):
        return self._phase

    @property
    def state(self):
        return self._state

    def set_amplitude(self, value, update=True):
        if value != self._amplitude and update:
            command = (self.channel, "amplitude", value)
            self._change_dds(command)
        self._amplitude = value

    def set_att(self, value, update=True):
        if value != self._att and update:
            command = (self.channel, "attenuation", value)
            self._change_dds(command)
        self._att = value

    def set_frequency(self, value, update=True):
        if value != self._frequency and update:
            command = (self.channel, "frequency", value)
            self._change_dds(command)
        self._frequency = value

    def set_phase(self, value, update=True):
        if value != self._phase and update:
            command = (self.channel, "phase", value)
            self._change_dds(command)
        self._phase = value

    def set_state(self, value, update=True):
        if value != self._state and update:
            if value:
                value_set = 1.0
            else:
                value_set = -1.0
            command = (self.channel, "state", value_set)
            self._change_dds(command)
        self._state = value

    def _change_dds(self, command):
        async def worker(self, command):
            await self.parent.artiq.set_dds(command)

        self.parent.run_in_labrad_loop(worker)(self, command)


class DDSDetail(DialogOnTop):
    """A dialog showing details for a channel."""

    def __init__(self, dds_parameters, parent=None):
        self.dds_parameters = dds_parameters
        super().__init__(parent)
        self.setWindowTitle(dds_parameters.channel)
        self.initialize_gui()
        self.setup_gui_listeners()

    def initialize_gui(self):
        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)
        labelfont = QtGui.QFont("Arial", 8)
        spinboxfont = QtGui.QFont("Arial", 10)

        label = QtWidgets.QLabel(f"CPLD: {self.dds_parameters.cpld}")
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        label.setFont(labelfont)
        grid.addWidget(label, 0, 0)

        label = QtWidgets.QLabel("Att (dB)")
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        label.setFont(labelfont)
        grid.addWidget(label, 1, 0)

        self.att_box = QtWidgets.QDoubleSpinBox()
        self.att_box.setDecimals(1)
        self.att_box.setMinimum(-31.5)
        self.att_box.setMaximum(0.0)
        self.att_box.setSingleStep(0.5)
        self.att_box.setFont(spinboxfont)
        self.att_box.setKeyboardTracking(False)
        self.att_box.setValue(-self.dds_parameters.att)
        grid.addWidget(self.att_box, 2, 0)

    def setup_gui_listeners(self):
        self.att_box.valueChanged.connect(self.on_widget_att_changed)

    def on_widget_att_changed(self, val):
        self.dds_parameters.set_att(-val)


class DDSChannel(QtWidgets.QGroupBox):
    """GUI for a DDS channel."""
    MHz_to_Hz = 1.0e6

    def __init__(self, dds_parameters, parent=None):
        self.dds_parameters = dds_parameters
        super().__init__(parent)
        self.initialize_gui()
        self.setup_gui_listeners()

    def initialize_gui(self):
        titlefont = QtGui.QFont("Arial", 10)
        labelfont = QtGui.QFont("Arial", 8)
        buttonfont = QtGui.QFont("Arial", 10)
        spinboxfont = QtGui.QFont("Arial", 10)

        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        label = QtWidgets.QLabel(self.dds_parameters.channel)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        label.setAlignment(QtCore.Qt.AlignHCenter)
        label.setFont(titlefont)
        grid.addWidget(label, 0, 0, 1, 3)

        label = QtWidgets.QLabel("Frequency (MHz)")
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        label.setFont(labelfont)
        grid.addWidget(label, 1, 0)

        label = QtWidgets.QLabel("Amplitude")
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        label.setFont(labelfont)
        grid.addWidget(label, 1, 1)

        self.freq_box = QtWidgets.QDoubleSpinBox()
        self.freq_box.setDecimals(3)
        self.freq_box.setMinimum(1.0)
        self.freq_box.setMaximum(500.0)
        self.freq_box.setSingleStep(0.1)
        self.freq_box.setFont(spinboxfont)
        self.freq_box.setKeyboardTracking(False)
        self.freq_box.setValue(self.dds_parameters.frequency / self.MHz_to_Hz)
        grid.addWidget(self.freq_box, 2, 0)

        self.amp_box = QtWidgets.QDoubleSpinBox()
        self.amp_box.setDecimals(5)
        self.amp_box.setMinimum(0.0)
        self.amp_box.setMaximum(1.0)
        self.amp_box.setSingleStep(0.01)
        self.amp_box.setFont(spinboxfont)
        self.amp_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        self.amp_box.setKeyboardTracking(False)
        self.amp_box.setValue(self.dds_parameters.amplitude)
        grid.addWidget(self.amp_box, 2, 1)

        self.switch_button = QtWidgets.QPushButton("o")
        self.switch_button.setFont(buttonfont)
        self.switch_button.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        self.switch_button.setCheckable(True)
        if self.dds_parameters.state:
            self.switch_button.setChecked(self.dds_parameters.state)
            self.set_switch_button_text(self.dds_parameters.state)
        grid.addWidget(self.switch_button, 2, 2)

    def setup_gui_listeners(self):
        self.freq_box.valueChanged.connect(self.on_widget_freq_changed)
        self.amp_box.valueChanged.connect(self.on_widget_amp_changed)
        self.switch_button.clicked.connect(self.on_widget_switch_changed)

    def on_widget_freq_changed(self, val):
        self.dds_parameters.set_frequency(val * self.MHz_to_Hz)

    def on_widget_amp_changed(self, val):
        self.dds_parameters.set_amplitude(val)

    def on_widget_switch_changed(self, checked):
        self.dds_parameters.set_state(checked)
        self.set_switch_button_text(checked)

    def on_monitor_freq_changed(self, val):
        self.freq_box.blockSignals(True)
        self.freq_box.setValue(val / self.MHz_to_Hz)
        self.freq_box.blockSignals(False)
        self.dds_parameters.set_frequency(val, False)

    def on_monitor_amp_changed(self, val):
        self.amp_box.blockSignals(True)
        self.amp_box.setValue(val)
        self.amp_box.blockSignals(False)
        self.dds_parameters.set_amplitude(val, False)

    def on_monitor_att_changed(self, val):
        self.dds_parameters.set_att(val, False)

    def on_monitor_switch_changed(self, checked):
        self.switch_button.blockSignals(True)
        self.switch_button.setChecked(checked)
        self.switch_button.blockSignals(False)
        self.set_switch_button_text(checked)
        self.dds_parameters.set_state(checked, False)

    def set_switch_button_text(self, checked):
        if checked:
            self.switch_button.setText("I")
        else:
            self.switch_button.setText("o")

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu()
        details_action = menu.addAction("Details")
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == details_action:
            self.show_details()

    def show_details(self):
        self.details = DDSDetail(self.dds_parameters, self)
        self.details.exec_()
