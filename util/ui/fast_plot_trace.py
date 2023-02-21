import numpy as _np
import pyqtgraph as _pg
from PyQt5 import QtCore


class _PlotPath(_pg.QtWidgets.QGraphicsPathItem):
    """Generates a path for x-y plotting data.

    Replaces the expensive work of plotting data by plotting a path. When plotting a large dataset,
    a path can be generated for every n data points. The path can be plotted instead of the
    n data points it replaces.
    """

    def __init__(self, x, y, pen):
        self.path = _pg.arrayToQPath(x.flatten(), y.flatten())
        super().__init__(self.path)
        self.setPen(pen)

    def shape(self):
        return super().shape()  # overriden because QGraphicsPathItem.shape is too expensive.

    def boundingRect(self):
        return self.path.boundingRect()


class FastPlotTrace(QtCore.QObject):
    """An interface to plot a trace efficiently.

    For a short trace (number of data points < length_to_path), it works the same as
    normal PyQtGraph data plotting. For a long trace, it saves every length_to_path elements
    as a _PlotPath object. The _PlotPath objects can be plotted more efficiently,
    and actual data plotted is short, so appending data is more efficient too.

    Args:
        plot_widget: pyqtgraph.PlotWidget to plot on.
        pen_kwargs: dict, pen arguments for the trace. See PyQtGraph documentation for details.
            Avoid using 'width' other than 1 for performance reasons.
        length_to_path: int, number of data points to save as a _PlotPath object.
    """
    # emitted when the trace is updated.
    # argument is the x-axis value of the last data point.
    trace_updated = QtCore.pyqtSignal(float)
    # emitted when the trace is removed.
    trace_removed = QtCore.pyqtSignal()

    def __init__(self, plot_widget, pen_kwargs={"color": "w", "width": 1},
                 length_to_path=20000):
        super().__init__()
        self._plot_widget = plot_widget
        self._pen = _pg.mkPen(**pen_kwargs)
        self._length_to_path = length_to_path
        self._plot_paths = []  # list of _PlotPath.
        self._plot_data_item = None  # actual PlotWidget.plot object.
        self._xdata = None
        self._ydata = None

    def update_trace(self, name=None):
        """Updates the trace on the plot."""
        if self._xdata is None or self._ydata is None:
            return
        # length of data to plot.
        # if xdata and ydata do not have the same length.
        # the longer one is not fully plotted.
        len_plot = min([len(self._xdata), len(self._ydata)])
        if len_plot == self._last_length:
            return
        last_x = self._xdata[-1]  # x-coordinate of the last data point.
        while len_plot > self._length_to_path:
            # creates _PlotPath until the length is shorter than self._length_to_path.
            path = _PlotPath(self._xdata[:self._length_to_path],
                             self._ydata[:self._length_to_path],
                             self._pen)
            self._plot_widget.addItem(path)
            self._plot_paths.append(path)
            self._xdata = self._xdata[self._length_to_path-1:]
            self._ydata = self._ydata[self._length_to_path-1:]
            len_plot = min([len(self._xdata), len(self._ydata)])
        if len_plot == len(self._xdata):
            xdata = self._xdata
        else:
            xdata = self._xdata[:len_plot]
        if len_plot == len(self._ydata):
            ydata = self._ydata
        else:
            ydata = self._ydata[:len_plot]
        if self._plot_data_item is not None:
            self._plot_data_item.setData(xdata, ydata)
        else:
            self._plot_data_item = self._plot_widget.plot(xdata, ydata, pen=self._pen, name=name)
        self._last_length = len_plot
        self.trace_updated.emit(last_x)

    def remove_trace(self):
        """Removes the trace from the plot."""
        for kk in self._plot_paths:
            self._plot_widget.removeItem(kk)
        self._plot_paths = []
        if self._plot_data_item is not None:
            self._plot_widget.removeItem(self._plot_data_item)
            self._plot_data_item = None
        self._last_length = -1
        self.trace_removed.emit()

    def set(self, xdata, ydata, name=None):
        """Sets data, clears the existing trace, and plots the new trace.

        Args:
            xdata: np.array, x-axis data.
            ydata: np.array, y-axis data.
        """
        self._xdata = _np.array(xdata)
        self._ydata = _np.array(ydata)
        self.remove_trace()
        self.update_trace(name)

    def append_x(self, xdata):
        """Appends to the x-axis data."""
        self._xdata = _np.append(self._xdata, _np.array(xdata), axis=0)

    def append_y(self, ydata):
        """Appends to the y-axis data."""
        self._ydata = _np.append(self._ydata, _np.array(ydata), axis=0)

    def append(self, xdata, ydata):
        """Appends to both x-axis and y-axis data, and updates the trace."""
        self.append_x(xdata)
        self.append_y(ydata)
        self.update_trace()
