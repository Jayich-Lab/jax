import argparse
import asyncio
import os
import threading
from PyQt5 import QtCore


__all__ = ["JaxApplet"]


class JaxApplet(QtCore.QObject):
    """Base class for all applets.
    
    This is modified from artiq.applets.simple.SimpleApplet.

    LabRAD cannot be run in the main thread as twisted asyncioreactor does not support the
    WindowsProactorEventLoop, which PyQt5 (qasync) requires.
    All LabRAD calls need to be done in a separate thread.
    """
    def __init__(self, cmd_description=None, **kwds):
        super().__init__(**kwds)
        self.argparser = argparse.ArgumentParser(description=cmd_description)
        self.argparser.add_argument("--ip", type=str, default="::1",
                                    help="LabRAD manager IP address to connect to")
        self.args = self.argparser.parse_args()
        self.init_window()

    #def run(self):
    #    self.show()

        #self.connect_labrad(self.args.ip)

    def init_window(self):
        if os.getenv("ARTIQ_APPLET_EMBED"):
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

    def connect_to_labrad(self, ip):
        """Connects to labrad in another thread (non-blocking).

        This function should be called by derived classes.
        After the connection finishes, self.labrad_connected will be called.
        """
        def worker():
            import selectors
            selector = selectors.SelectSelector()
            self._labrad_loop = asyncio.SelectorEventLoop(selector)
            asyncio.set_event_loop(self._labrad_loop)

            from twisted.internet import asyncioreactor
            asyncioreactor.install(self._labrad_loop)
            self._labrad_loop.create_task(self.labrad_worker(ip))
            self._labrad_loop.run_forever()

        self._labrad_thread = threading.Thread(target=worker)
        self._labrad_thread.start()

    async def labrad_worker(self, ip):
        """Worker to connect to labrad in self._labrad_loop event loop.

        To make multiple labrad connections, this function should be overridden.
        """
        from pydux.lib.control.clients.connection_asyncio import ConnectionAsyncio
        self.cxn = ConnectionAsyncio()
        await self.cxn.connect(ip)
        await self.labrad_connected()
        while True:
            await asyncio.sleep(0.)

    async def labrad_connected(self):
        """Called when the labrad connection self.cxn is set.

        Should be implemented by derived classes.
        """
        raise NotImplementedError("This function must be overriden by derived classes.")

    def run_in_labrad_loop(self, func):
        """Wrapper for an async function to run in self._labrad_loop.

        All code that uses labrad needs to be run in the labrad event loop.

        Args:
            func: async function to control labrad.

        Returns:
            func_ensured_future: sync function.
        """
        def func_ensured_future(*args, **kwargs):
            asyncio.ensure_future(func(*args, **kwargs), loop=self._labrad_loop)

        return func_ensured_future

    def closeEvent(self, event):
        if self._labrad_loop is not None:
            self._labrad_loop.stop()
