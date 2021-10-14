import asyncio
import threading
from PyQt5 import QtCore


__all__ = ["JaxApplet"]


class JaxApplet(QtCore.QObject):
    """Base class for all applets.

    LabRAD cannot be run in the main thread as twisted asyncioreactor does not support the
    WindowsProactorEventLoop, which ARTIQ requires. All LabRAD calls need to be done in a
    separate thread.
    """
    @classmethod
    def add_labrad_ip_argument(self, applet, default_ip="127.0.0.1"):
        """Adds an argument to set the LabRAD IP address to connect to.

        Args:
            applet: artiq.applets.simple.SimpleApplet object.
            default_ip: str, default IP addres to connect to. Default "127.0.0.1" (local computer).
        """
        applet.argparser.add_argument("--ip", type=str, default=default_ip,
                                      help="LabRAD manager IP address to connect to")

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._labrad_loop = None

    def connect_to_labrad(self, ip="127.0.0.1"):
        """Connects to labrad in another thread (non-blocking).

        This function should be called by derived classes.
        After the connection finishes, self.labrad_connected will be called.
        The event loop in artiq.applets.simple.SimpleApplet is not compatible with asyncioreactor.
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

    def data_changed(self, data, mods):
        """We don't use ARTIQ dataset manager so this function is not used."""
        pass

    async def labrad_worker(self, ip):
        """Worker to connect to labrad in self._labrad_loop event loop."""
        from pydux.lib.control.clients.connection_asyncio import ConnectionAsyncio
        self.cxn = ConnectionAsyncio()
        await self.cxn.connect(ip)
        await self.labrad_connected()
        while True:  # required for the event loop to keep handling events.
            short_time = 0.01
            await asyncio.sleep(short_time)

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