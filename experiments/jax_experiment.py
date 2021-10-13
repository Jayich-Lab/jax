from artiq.experiment import Experiment


class JaxExperiment(Experiment):
    """Base class for all Jayich lab experiments."""
    def build(self):
        super().build()

    def prepare(self):
        """Skips Experiment.prepare() and goes to prepare() defined in the environment."""
        delattr(Experiment, "prepare")
        super().prepare()
