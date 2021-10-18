import numpy as _np
from labrad.units import WithUnit, WithDimensionlessUnit


__all__ = ["remove_labrad_units"]


def remove_labrad_units(value):
    """Removes labrad unit from returned value of a labrad server.

    Labrad sends some types of values in WithUnit or WithDimensionlessUnit types,
    which we need to convert to np.array to serialize them or use them in ARTIQ python.

    Tuple and list objects needs to be iterated over to remove labrad unit types.
    WithUnit is converted to the base unit first, and then converted to np.array.
    """
    def labrad_type_to_array(value):
        if isinstance(value, (WithDimensionlessUnit, WithUnit)):
            value = _np.array(value.inBaseUnits())
            if _np.ndim(value) == 0:
                value = value.item()  # sipyco.pyon interprets numpy scalars as 1d arrays.
        return value

    if isinstance(value, (tuple, list)):
        new_value = []
        for kk in value:
            new_value.append(remove_labrad_units(kk))
        if isinstance(value, tuple):
            new_value = tuple(new_value)
        return new_value
    return labrad_type_to_array(value)
