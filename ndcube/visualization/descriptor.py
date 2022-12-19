import functools

MISSING_MATPLOTLIB_ERROR_MSG = ("Matplotlib can not be imported, so the default plotting "
                                "functionality is disabled. Please install matplotlib.")


class PlotterDescriptor:
    def __init__(self, default_type=None):
        self._default_type = default_type

    def __set_name__(self, owner, name):
        """
        This function is called when the class the descriptor is attached to is initialized.

        The *class* and not the instance.
        """
        # property name is the name of the attribute on the parent class
        # pointing at an instance of this descriptor.
        self._property_name = name
        # attribute name is the name of the attribute on the parent class where
        # the data is stored.
        self._attribute_name = f"_{name}"
        plotter = self._resolve_default_type(raise_error=False)
        if plotter is not None and hasattr(plotter, "plot"):
            functools.update_wrapper(owner.plot, plotter.plot)

    def _resolve_default_type(self, raise_error=True):
        # We special case the default MatplotlibPlotter so that we can
        # delay the import of matplotlib until the plotter is first
        # accessed.
        if self._default_type in ("mpl_plotter", "mpl_sequence_plotter"):
            try:
                if self._default_type == "mpl_plotter":
                    from ndcube.visualization.mpl_plotter import MatplotlibPlotter
                    return MatplotlibPlotter
                elif self._default_type == "mpl_sequence_plotter":
                    from ndcube.visualization.mpl_sequence_plotter import MatplotlibSequencePlotter
                    return MatplotlibSequencePlotter
            except ImportError as e:
                if raise_error:
                    raise ImportError(MISSING_MATPLOTLIB_ERROR_MSG) from e

        elif self._default_type is not None:
            return self._default_type

        # If we have no default type then just return None
        else:
            return

    def __get__(self, obj, objtype=None):
        if obj is None:
            return

        if getattr(obj, self._attribute_name, None) is None:
            plotter_type = self._resolve_default_type()
            if plotter_type is None:
                return

            self.__set__(obj, plotter_type)

        return getattr(obj, self._attribute_name)

    def __set__(self, obj, value):
        if not isinstance(value, type):
            raise TypeError(
                "Plotter attribute can only be set with an uninitialised plotter object.")

        setattr(obj, self._attribute_name, value(obj))
        # here obj is the ndcube object and value is the plotter type
        # Get the instantiated plotter we just assigned to the ndcube
        plotter = getattr(obj, self._attribute_name)
        # If the plotter has a plot object then update the signature and
        # docstring of the cubes `plot()` method to match
        # Docstrings of methods aren't writeable so we copy to the underlying
        # function object instead
        if hasattr(plotter, "plot"):
            functools.update_wrapper(obj.plot.__func__, plotter.plot.__func__)
