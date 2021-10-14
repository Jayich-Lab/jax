from jax.util.tree_dict import TreeDict


__all__ = ["ParameterGroup"]


class ParameterGroup(TreeDict):
    """Holds parameters for an experiment in an ARTIQ python compatible class.

    Args:
        parameters_dict: A nested dict representing the parameters. Example:
            {
                "collection_1": {
                    "parameter_1": value_1,
                    "parameter_2": value_2
                },
                "collection_2": {
                    "parameter_3": value_3
                }
            }.
    """
    def __init__(self, parameters_dict):
        super().__init__(parameters_dict)
