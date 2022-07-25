from artiq.experiment import *
from jax import JaxExperiment, SinaraEnvironment
from jax.examples.sequences.example2 import Example2
from jax.util.parameter_group import ParameterGroup
from jax.util.drift_tracker import DriftTracker


__all__ = ["PulseSequence"]


class PulseSequence(JaxExperiment, SinaraEnvironment):
    """Example experiment that contains a pulse sequence and demonstrates data saving.

    This is an experiment that runs a Doppler cooling - state detection sequence repeatedly.
    The experiment can be cleanly terminated by the user, and it can handover to another higher
    priority experiment too.

    To run this experiment, you need to run the "artiq" and "vault" labrad servers.
    It also demonstrates how to use "parameter_bank" and "drift_tracker" servers with an
    experiment. The parameter_bank server saves parameters of the experiment, and the
    drift_tracker server provides drift trackers for ion transition frequencies.
    Set USE_PARAMETER_BANK and/or USE_DRIFT_TRACKER to True if the user
    wants to use these servers. Otherwise, mock parameters and drift trackers are used.

    Before running this experiment, the DDSes should be terminated with 50 ohm terminators.
    To run this experiment, import this experiment class in a file of the repository that
    artiq_master controls (see ARTIQ manual), and the experiment should show up after
    "scanning repository HEAD" using the experiment explorer in the artiq dashboard.
    """
    USE_PARAMETER_BANK = False
    USE_DRIFT_TRACKER = False

    def build(self):
        super().build()
        # parameters required by the experiment.
        exp_parameters = [
            ("example2", "num_of_repeats"),
            ("example2", "cool_time"),
            ("example2", "wait_time")
        ]
        self.get_parameter_paths(exp_parameters, [Example2])  # populates self.parameter_paths

    def prepare(self):
        super().prepare()
        if self.USE_PARAMETER_BANK:
            self.save_parameters()
        else:
            self.get_mock_parameters()
        if not self.USE_DRIFT_TRACKER:
            self.get_mock_drift_trackers()

        self.sequence = Example2(self, self.p, self.p.example2.cool_time, self.p.example2.wait_time)

    def run(self):
        try:
            self.repeats_done = 0  # tracks how many repeatitions have been done.
            self.open_file()  # opens up a file for writing data.
            # defines a instance variable that will be used in the kernel.
            # instance variables cannot be defined in the kernel, but can be modified
            # the variable's type is not changed.
            self.counts_dset_name = ""
            while self.repeats_done < self.p.example2.num_of_repeats:
                # checks if user has stopped the experiment.
                should_stop = self.check_stop_or_do_pause()
                if should_stop:
                    break
                else:
                    self.turn_off_all_ddses()
                    self.run_kernel()
        except Exception as e:
            raise e
        finally:
            self.reset_sinara_hardware()  # resets the hardware to pre-experiment state.
            self.close_file()  # closes the data file.
            self.disconnect_labrad()  # closes the labrad connection.

    def get_mock_parameters(self):
        """Change 'devices' section to set to valid device names."""
        params = {
            "example2": {
                "num_of_repeats": 1000,
                "cool_time": 5*ms,
                "wait_time": 5*us
            },
            "devices": {
                "cool_dds": "dp_468",
                "repump_dds": "dp_1079",
                "pmt_edge_counter": "pmt_counter"
            },
            "state_detect": {
                "cool_detuning": -20*MHz,
                "cool_amplitude": 0.1,
                "cool_drift_tracker": "468",
                "repump_detuning": 20*MHz,
                "repump_amplitude": 0.1,
                "repump_drift_tracker": "1079",
                "detect_time": 1*ms
            },
            "doppler_cool": {
                "cool_detuning": -20*MHz,
                "cool_amplitude": 0.05,
                "cool_drift_tracker": "468",
                "repump_detuning": 20*MHz,
                "repump_amplitude": 0.05,
                "repump_drift_tracker": "1079"
            }
        }
        self.p = ParameterGroup(params)
        self.add_attribute("parameters", self.serialize(params))

    def get_mock_drift_trackers(self):
        self.drift_trackers = {}
        param_468 = {"center_frequency": 260*MHz, "detuning_factor": -2, "center_drift_rate": 0.,
                     "last_calibration": 0., "Zeeman": None}
        self.drift_trackers["468"] = DriftTracker(param_468)
        self.add_attribute("468", self.serialize(param_468), "drift_trackers")
        param_1079 = {"center_frequency": 105*MHz, "detuning_factor": -2, "center_drift_rate": 0.,
                      "last_calibration": 0., "Zeeman": None}
        self.drift_trackers["1079"] = DriftTracker(param_1079)
        self.add_attribute("1079", self.serialize(param_1079), "drift_trackers")

    @kernel
    def run_kernel(self):
        self.core.reset()
        while self.repeats_done < self.p.example2.num_of_repeats:
            # if the experiment should pause or stop. This function takes several ms to run.
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()
            count = self.sequence.run()  # runs the pulse sequence.
            if self.repeats_done == 0:
                # initializes a dataset.
                self.counts_dset_name = self.add_dataset("counts", [count])
            else:
                # appends to the dataset.
                self.append_dataset(self.counts_dset_name, [count])
            self.repeats_done += 1
