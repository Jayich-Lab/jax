import asyncio
import threading
import math
import vispy.scene
from vispy import gloo, app, visuals
from PyQt5 import QtCore, QtGui, QtWidgets
from artiq.applets.simple import SimpleApplet
from jax import JaxApplet
import numpy as np
import vispy.plot as vp
from time import time
app.use_app("pyqt5")
#__all__ = ["VispyApplet"]



class VispyApplet(QtWidgets.QDialog):
    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self.initialize_gui()
        asyncio.ensure_future(self.run())

    async def run(self):
        while True:
            print(time() - self.time)
            self.time = time()
            await asyncio.sleep(0.001)
            self.data = np.append(self.data, np.random.rand(1, 2) + np.array([len(self.data),0]), axis=0)
            self.line1.set_data(self.data)

    def data_changed(self, *args, **kwargs):
        pass

    def initialize_gui(self):
        self.time = time()
        self.data = np.random.rand(10000, 2) # random data
        self.fig = vp.Fig(size=(800, 600))
        self.line1 = self.fig[0, 0].plot(data=self.data, marker_size=0.)

        # PyQt (with vispy fig.native)
        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self.fig.native)
        self.fig.show()
        self.setLayout(layout)





def main():
    applet = SimpleApplet(VispyApplet)
    applet.run()


if __name__ == "__main__":
    main()
