import numpy as _np
from PyQt5 import QtWidgets
from jax import RealTimePlotApplet


class PlotPMT(RealTimePlotApplet):
    def __init__(self, **kwds):
        num_of_traces = 3  # high, low, differential.
        datasets_names = ["pmt.times", "pmt.counts_kHz"]
        xlabel = "Time (s)"
        ylabel = "Counts (kHz)"
        super().__init__(num_of_traces, datasets_names, xlabel, ylabel, **kwds)

    async def initialize_datasets(self):
        try:
            times = await self.dv.get_shared_dataset("pmt.times")
            counts = await self.dv.get_shared_dataset("pmt.counts_kHz")
        except Exception as e:
            print(e)
            times = _np.array([])
            counts = _np.array([[], [], []])
        self.set_data.emit("pmt.times", times)
        self.set_data.emit("pmt.counts_kHz", counts)

    def _set(self, dataset_name, value):
        if dataset_name == "pmt.times":
            self._times = value
        elif dataset_name == "pmt.counts_kHz":
            value = _np.transpose(value)
            for kk in range(len(value)):
                self.traces[kk].set(self._times, value[kk])

    def _append(self, dataset_name, value):
        if dataset_name == "pmt.times":
            for kk in range(len(self.traces)):
                self.traces[kk].append_x(value)
        elif dataset_name == "pmt.counts_kHz":
            value = _np.transpose(value)
            for kk in range(len(self.traces)):
                self.traces[kk].append_y(value[kk])
                self.traces[kk].update_trace()  # only update the plot when counts are updated.


def main():
    import asyncio
    from qasync import QEventLoop
    app = QtWidgets.QApplication([])
    loop = QEventLoop()
    asyncio.set_event_loop(loop)
    applet = PlotPMT()
    applet.show()
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
