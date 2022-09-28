import asyncio

import numpy as _np
import pyqtgraph as _pg
from jax import JaxApplet
from jax.util.ui.fast_plot_trace import FastPlotTrace
from PyQt5 import QtCore, QtGui, QtWidgets

__all__ = ["RealTimePlotApplet"]


class RealTimePlotApplet(QtWidgets.QWidget, JaxApplet):
    """Base applet for making a real-time updated plot.

    Plots shared datasets in the vault server.
    A plot can contain multiple traces (FastPlotTrace), stored in self.traces.
    When the control key is pressed down, automatic scrolling is disabled.

    Derived classes should implement self.initialize_datasets, self._set, and self._append.

    Args:
        num_of_traces: int, total number of traces to plot in the figure.
        dataset_names: list of strs, names of shared dataset to get updates for.
        xlabel: str, x-axis label. Default "".
        ylabel: str, y-axis label. Default "".
        scrolling: bool, whether the viewport scrolls with new data. Default True.
        ip: str, vault server IP address to connect to.
        ylog: bool, determined whether the y axis will be plotting logarithmically
    """

    set_data = QtCore.pyqtSignal(str, _np.ndarray)
    append_data = QtCore.pyqtSignal(str, _np.ndarray)

    def __init__(
        self,
        num_of_traces,
        dataset_names,
        xlabel="",
        ylabel="",
        scrolling=True,
        ip="127.0.0.1",
        ylog=False,
        **kwds,
    ):
        super().__init__(**kwds)
        self.num_of_traces = num_of_traces
        self.dataset_names = dataset_names
        self.ylog = ylog
        self.scrolling = scrolling
        self._control_pressed = False
        self.setDisabled(True)
        self._color_index = 0
        self._initialize_gui(xlabel, ylabel)

        self.set_data.connect(self._set)
        self.append_data.connect(self._append)
        self.connect_to_labrad(ip)

    def _initialize_gui(self, xlabel, ylabel):
        layout = QtWidgets.QGridLayout(self)
        self.plot_widget = _pg.PlotWidget()
        self._set_axes_style(xlabel, ylabel)
        layout.addWidget(self.plot_widget, 0, 0)
        self.coords = QtWidgets.QLabel("")
        self.coords.setFont(QtGui.QFont("Arial", 15))
        layout.addWidget(self.coords, 1, 0)
        self.setLayout(layout)
        self._set_all_traces()

        self.plot_widget.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot_widget.sigRangeChanged.connect(self._range_changed)

    def _set_axes_style(self, xlabel, ylabel):
        axis_fontsize = 15
        tick_text_offset = 10
        y_width = 90
        label_style = {"color": "#AAA", "font-size": f"{axis_fontsize}pt"}
        font = QtGui.QFont("Arial", axis_fontsize)

        x = self.plot_widget.plotItem.getAxis("bottom")
        x.setLabel(xlabel, **label_style)
        x.setStyle(tickFont=font, tickTextOffset=tick_text_offset)

        if self.ylog:
            self.plot_widget.plotItem.setLogMode(y=True)
        y = self.plot_widget.plotItem.getAxis("left")
        y.setLabel(ylabel, **label_style)
        y.setStyle(tickFont=font, tickTextOffset=tick_text_offset)
        y.setWidth(y_width)

    def _set_all_traces(self):
        """Adds all traces with default colors."""
        self.traces = []
        for kk in range(self.num_of_traces):
            self.traces.append(self.new_trace())

    def new_trace(self, color=None):
        """Creates a new trace.

        Always use trace width of 1 if possible. Non-unity trace width reduces plot speed
        significantly. This may be fixed soon (see pyqtgraph PR #2011).

        Args:
            color: color of the trace. See pyqtgraph color documention for details.
                If None, a color from a color wheel will be used.
        """
        color_wheel = ["w", "c", "y", "g", "r", "m"]
        if color is None:
            color = color_wheel[self._color_index % len(color_wheel)]
            self._color_index += 1
        trace = FastPlotTrace(self.plot_widget, pen_kwargs={"color": color, "width": 1})
        trace.trace_updated.connect(self._trace_updated)
        trace.trace_removed.connect(self._trace_removed)
        return trace

    def _mouse_moved(self, position):
        if self.plot_widget.sceneBoundingRect().contains(position):
            point = self.plot_widget.plotItem.vb.mapSceneToView(position)
            coordinates = f"({point.x():.8}, {point.y():.8})"
            self.coords.setText(coordinates)

    def _range_changed(self):
        lims = self.plot_widget.viewRange()
        self.current_xlimits = [lims[0][0], lims[0][1]]

    def keyPressEvent(self, key_event):
        if key_event.key() == QtCore.Qt.Key_Control:
            self._control_pressed = True
        super().keyPressEvent(key_event)

    def keyReleaseEvent(self, key_event):
        if key_event.key() == QtCore.Qt.Key_Control:
            self._control_pressed = False
        super().keyPressEvent(key_event)

    def _trace_updated(self, data_xmax):
        # it is probably easier to disable scrolling by pressing the mouse.
        # however, PyQtGraph does not emit the MouseReleaseEvent.
        # so the control key is used instead.
        if self.scrolling and not self._control_pressed:
            try:
                plot_xmin, plot_xmax = self.current_xlimits
                plot_width = plot_xmax - plot_xmin
                # scroll if we have reached 80% of the window
                if data_xmax > (plot_xmin + 0.8 * plot_width) and data_xmax < plot_xmax:
                    shift = plot_width / 4
                    xmin = plot_xmin + shift
                    xmax = plot_xmax + shift
                    self.plot_widget.setXRange(xmin, xmax)
                    self.current_xlimits = [xmin, xmax]
            except Exception as e:
                pass

    def _trace_removed(self):
        pass  # does not need to do anything when a trace is removed.

    async def labrad_connected(self):
        await self.vault_connected()
        await self.setup_cxn_listeners()

    async def vault_connected(self):
        self.dv = self.cxn.get_server("vault")
        for kk in self.dataset_names:
            await self.dv.subscribe_to_shared_dataset(kk)

        try:
            await self.initialize_datasets()  # implemented by the derived class.
        except Exception as e:
            print(e)

        SHARED_DATA_CHANGE = 128936
        await self.dv.on_shared_data_change(SHARED_DATA_CHANGE)
        self.dv.addListener(
            listener=self._data_changed, source=None, ID=SHARED_DATA_CHANGE
        )
        self.setDisabled(False)

    async def setup_cxn_listeners(self):
        self.cxn.add_on_connect("vault", self.run_in_labrad_loop(self.vault_connected))
        self.cxn.add_on_disconnect("vault", self.vault_disconnected)

    def vault_disconnected(self):
        self.setDisabled(True)

    def _data_changed(self, signal, data):
        operation = data[0]
        dataset_name = data[1]
        value = data[2]
        if operation == "set":
            self.set_data.emit(dataset_name, value)
        elif operation == "append":
            self.append_data.emit(dataset_name, value)

    async def initialize_datasets(self):
        """Implement this function to get initial dataset values."""
        raise NotImplementedError()

    def _set(self, dataset_name, value):
        """Called when a dataset listened to is set to a new value."""
        raise NotImplementedError()

    def _append(self, dataset_name, value):
        """Called when a dataset listened to is appended by a new value."""
        raise NotImplementedError()
