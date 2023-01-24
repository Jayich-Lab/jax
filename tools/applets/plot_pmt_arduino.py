import numpy as _np
from artiq.applets.simple import SimpleApplet
from jax import RealTimePlotApplet


class PlotPMTArduino(RealTimePlotApplet):
    def __init__(self, args, **kwds):
        num_of_traces = 3  # high, low, differential.
        datasets_names = ["pmt_arduino.times", "pmt_arduino.counts_kHz"]
        xlabel = "Time (s)"
        ylabel = "Counts (kHz)"
        super().__init__(
            num_of_traces, datasets_names, xlabel, ylabel, ip=args.ip, **kwds
        )

    async def initialize_datasets(self):
        try:
            times = await self.dv.get_shared_dataset("pmt_arduino.times")
            counts = await self.dv.get_shared_dataset("pmt_arduino.counts_kHz")
        except Exception as e:
            print(e)
            times = _np.array([])
            counts = _np.array([[], [], []])
        self.set_data.emit("pmt_arduino.times", times)
        self.set_data.emit("pmt_arduino.counts_kHz", counts)

    def _set(self, dataset_name, value):
        if dataset_name == "pmt_arduino.times":
            self._times = value
        elif dataset_name == "pmt_arduino.counts_kHz":
            value = _np.transpose(value)
            for kk in range(len(value)):
                self.traces[kk].set(self._times, value[kk])

    def _append(self, dataset_name, value):
        if dataset_name == "pmt_arduino.times":
            for kk in range(len(self.traces)):
                self.traces[kk].append_x(value)
        elif dataset_name == "pmt_arduino.counts_kHz":
            value = _np.transpose(value)
            for kk in range(len(self.traces)):
                self.traces[kk].append_y(value[kk])
                self.traces[
                    kk
                ].update_trace()  # only update the plot when counts are updated.


def main():
    applet = SimpleApplet(PlotPMTArduino)
    PlotPMTArduino.add_labrad_ip_argument(applet)
    applet.run()


if __name__ == "__main__":
    main()
