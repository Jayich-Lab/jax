import numpy as _np
import pyqtgraph as _pg
from PyQt5 import QtCore


class PlotPath(_pg.QtGui.QGraphicsPathItem):
    def __init__(self, x, y, pen):
        # TODO: check finite performance. , finiteCheck=False
        self.path = _pg.arrayToQPath(x.flatten(), y.flatten())
        _pg.QtGui.QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pen)

    def shape(self): # override because QGraphicsPathItem.shape is too expensive.
        return _pg.QtGui.QGraphicsItem.shape(self)

    def boundingRect(self):
        return self.path.boundingRect()


class PlotDataset(QtCore.QObject):
    def __init__(self, plot_widget, pen_kwargs={"color": "w", "width": 1},
                 length_to_path=20000):
        self._plot_widget = plot_widget
        self._pen = _pg.mkPen(**pen_kwargs)
        self._length_to_path = length_to_path
        self._plot_paths = []
        self._plot_data_item = None
        self._xdata = None
        self._ydata = None

    def update_plot(self):
        len_plot = min([len(self._xdata), len(self._ydata)])
        if len_plot == self._last_length:
            return
        while len_plot > self._length_to_path:
            path = PlotPath(self._xdata[:self._length_to_path],
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
            self._plot_data_item = self._plot_widget.plot(xdata, ydata, pen=self._pen)
        self._last_length = len_plot

    def remove_plot(self):
        for kk in self._plot_paths:
            self._plot_widget.removeItem(kk)
        self._plot_paths = []
        if self._plot_data_item is not None:
            self._plot_widget.removeItem(self._plot_data_item)
            self._plot_data_item = None
        self._last_length = -1

    def set(self, xdata, ydata):
        self._xdata = _np.array(xdata)
        self._ydata = _np.array(ydata)
        self.remove_plot()
        self.update_plot()

    def append(self, xdata, ydata):
        self._xdata = _np.append(self._xdata, xdata, axis=0)
        self._ydata = _np.append(self._ydata, ydata, axis=0)
        self.update_plot()
