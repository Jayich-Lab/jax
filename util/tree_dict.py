__all__ = ["TreeDict"]


class TreeDict:
    """Converts a nested dict to an object.

    Items in the dict are set to object attributes.
    ARTIQ python does not support dict type. Inherit this class to convert the dict to an object.
    self.value_parser() can be inherited to parse non-dict values.

    Args:
        dict_value: dict, dictionary to convert to an object.
        nested_dict_class: class for nested dicts. Default None, which represents self.__class__.
            This can be a different class (usually another class inherited from TreeDict).
    """
    def __init__(self, dict_value, nested_dict_class=None):
        self._set_attributes(dict_value, nested_dict_class)

    def value_parser(self, value):
        """Parser for non-dict values."""
        return value

    def _set_attributes(self, dict_value, nested_dict_class):
        if nested_dict_class is None:
            class SubClass(self.__class__):
                """A derived class from the current class.

                ARTIQ python does not support nesting a class as an attribute of the same class,
                so a derived class from self.__class__ is necessary.
                """
                pass

            nested_dict_class = SubClass
        for item in dict_value:
            if isinstance(dict_value[item], dict):
                setattr(self, item, nested_dict_class(dict_value[item]))
            else:
                setattr(self, item, self.value_parser(dict_value[item]))
