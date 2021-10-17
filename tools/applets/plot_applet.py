import asyncio
import numpy as _np
import pyqtgraph as _pg
from PyQt5 import QtCore, QtGui, QtWidgets
from artiq.applets.simple import SimpleApplet
from jax import JaxApplet
from jax.util.plot_dataset import PlotDataset


class PlotApplet(QtWidgets.QWidget, JaxApplet):
    update_plot = QtCore.pyqtSignal(object)
    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self.update_plot.connect(self._update_plot)
        self.initialize_gui()
        self.connect_to_labrad()

    async def labrad_connected(self):
        while True:
            await asyncio.sleep(0.01)
            data = _np.random.rand(100)
            self.update_plot.emit(data)

    def _update_plot(self, data):
        if self.counter > 100000:
            self.counter = 0
            self.plot_dataset.set(self.counter + _np.arange(len(data)), data)
        else:
            self.plot_dataset.append(self.counter + _np.arange(len(data)), data)
            self.counter += len(data)

    def initialize_gui(self):
        layout = QtWidgets.QGridLayout(self)
        self.plot_widget = _pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        self.plot_dataset = PlotDataset(self.plot_widget)
        self.counter = 0
        data = _np.random.rand(100000)
        self.plot_dataset.set(self.counter + _np.arange(len(data)), data)
        self.counter += len(data)


def main():
    applet = SimpleApplet(PlotApplet)
    applet.run()


if __name__ == "__main__":
    main()
