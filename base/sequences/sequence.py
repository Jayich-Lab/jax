from artiq.experiment import *


class Sequence:
    """Base class for pulse sequences.

    Pulse sequences are building blocks of kernel code.
    A sequence can contain other Sequence instances as subsequences.

    Set required_parameters to a list of parameters used in the sequence.
    Set required_subsequences to a list of sequences used in the sequence.
    They must be fully populated before __init__().

    Args:
        exp: experiment instance.
        parameter_group: ParameterGroup, experiment parameters.
    """
    required_parameters = []
    required_subsequences = []

    @classmethod
    def all_required_parameters(cls):
        """Returns all required parameters in the sequence and its subsequences."""
        parameters = list(cls.required_parameters)
        for kk in cls.required_subsequences:
            parameters.extend(kk.all_required_parameters())
        return list(set(parameters))

    def __init__(self, exp, parameter_group):
        self.exp = exp
        self.p = parameter_group

    @kernel
    def run(self):
        """Override this function to construct the pulse sequence."""
        pass
