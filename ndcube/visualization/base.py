import abc


class BasePlotter(abc.ABC):
    """
    Base class for NDCube plotter objects.
    """

    def __init__(self, ndcube=None):
        self._ndcube = ndcube

    @abc.abstractmethod
    def plot(self, *args, **kwargs):
        """
        The default plot method.

        ``Plotter`` classes should provide a ``plot()`` method which is called
        when users access `.NDCube.plot`. It should strive to provide a good
        overview of the cube by default but the behaviour is left to the
        implementation.

        The ``plot()`` method **should** accept ``**kwargs``.
        """
