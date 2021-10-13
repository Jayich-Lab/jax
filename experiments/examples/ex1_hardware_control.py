# It is important to import this base experiment module instead of the JaxExperiment class.
# The JaxExperiment class is derived from artiq.experiment.Experiment, and artiq will try to
# load any derived classes from artiq.experiment.Experiment into the experiment explorer.
# Since JaxExperiment and Experiment classes don't have the __init__ method, they cannot be
# successfully loaded into the experiment explorer, and will raise an error.
import jax.experiments.jax_experiment as _jax_exp
from artiq.experiment import *
from jax.experiments.sinara_environment import SinaraEnvironment


class HardwareControl(_jax_exp.JaxExperiment, SinaraEnvironment):
    """Example experiment controlling a DDS and a TTL.

    An experiment must first inherit from an base experiment and then inherit an environment.
    The base experiment most importantly defines the self.run() function. It may contain
    scan structures, and it may use pulse sequences.
    The environment sets up the labrad connection and provides functions for data saving,
    loading parameters, resetting hardware, etc.

    This is an simple experiment changing a DDS and a TTL, and reset them at the end of
    the experiment. To run this experiment, you need to run the "artiq" labrad server.
    Device names in self.build() can be changed, with the names in self.run_kernel()
    changed correspondingly.

    Before running this experiment, the DDS output should be terminated with a 50 ohm terminator.
    To run this experiment, import this experiment class in a file of the repository that
    artiq_master controls (see ARTIQ manual), and the experiment should show up after
    "scanning repository HEAD" using the experiment explorer in the artiq dashboard.
    """
    def build(self):
        super().build()  # Calls JaxExperiment.build(), which calls SinaraEnvironment.build()
        self.setattr_device("dp_468")  # a AD9910 DDS
        self.setattr_device("ttl4")  # a TTL output

    def prepare(self):
        super().prepare()  # Calls JaxExperiment.prepare(), which calls SinaraEnvironment.prepare()

    def run(self):
        try:
            self.turn_off_all_ddses()
            self.run_kernel()  # Runs code on the device.
        except Exception as e:
            raise e
        finally:
            self.reset_sinara_hardware()  # resets the hardware to pre-experiment state.
            self.disconnect_labrad()  # closes the labrad connection.

    @kernel
    def run_kernel(self):
        self.core.reset()  # resets the core and clears FIFOs.
        self.core.break_realtime()
        self.dp_468.set_att(15.)  # sets DDS attenuation.
        self.core.break_realtime()
        delay_mu(self.dds_set_delay_mu)
        self.dp_468.set(300*MHz, 0., 0.1)  # sets a DDS.
        self.core.break_realtime()
        self.dp_468.sw.on()  # turns on a DDS.
        self.core.break_realtime()
        self.ttl4.on()  # turns on a TTL.
