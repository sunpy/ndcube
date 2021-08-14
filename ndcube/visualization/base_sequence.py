import abc


class BaseSequencePlotter(abc.ABC):
    """
    Base class for NDCube plotter objects.
    """

    def __init__(self, ndcubesequence=None):
        self._sequence = ndcubesequence

    @abc.abstractmethod
    def plot(self, *args, **kwargs):
        """
        The default plot method.

        ``Plotter`` classes should provide a ``plot()`` method which is called
        when users access `.NDCubeSequence.plot`. It should strive to provide a good
        overview of the sequence by default but the behaviour is left to the
        implementation.

        The ``plot()`` method **should** accept ``**kwargs``.
        """
