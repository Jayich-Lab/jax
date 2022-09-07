import numpy as _np
from artiq.experiment import TInt64, kernel
from jax import JaxExperiment

__all__ = ["InfiniteLoop"]


class InfiniteLoop(JaxExperiment):
    """Base class for all infinite loop experiments.

    The sequence during self.run() is shown below:
        self.host_startup()  # host code to set up the experiment.
        self.kernel_run()  # run the following indented kernel functions.
            self.kernel_before_loops()  # kernel code to set up the loop.
                # checks if the experiment should be terminated or paused.
                self.kernel_loop()  # runs a loop in kernel code.
            self.kernel_after_loops()  # kernel code to clean up the loop.
        self.host_cleanup()  # host code to clean up the experiment.

    Use RPCs to call host functions during the kernel execution if needed.
    """

    def run(self):
        try:
            self.host_startup()
            while True:
                # blocks if a higher priority experiment takes control.
                if self.check_stop_or_do_pause():
                    # if termination is requested.
                    break
                else:
                    self.turn_off_all_ddses()
                    self.kernel_run()
        except Exception as e:
            raise e
        finally:
            self.host_cleanup()

    def host_startup(self):
        """Called at the start of self.run(). Can be overriden."""
        self.open_file()

    def host_cleanup(self):
        """Called at the end of self.run(). Can be overriden."""
        self.reset_sinara_hardware()
        self.close_file()
        self.disconnect_labrad()

    @kernel
    def kernel_run(self):
        self.kernel_before_loops()
        loop_index = _np.int64(0)
        while True:
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()
            self.kernel_loop(loop_index)
        self.kernel_after_loops()

    @kernel
    def kernel_before_loops(self):
        """Called at the start of self.kernel_run(). Can be overriden."""
        self.core.reset()

    @kernel
    def kernel_loop(self, loop_index: TInt64):
        """Called during each loop of self.kernel_run(). Can be overriden."""
        pass

    @kernel
    def kernel_after_loops(self):
        """Called at the end of self.kernel_run(). Can be overriden."""
        pass
