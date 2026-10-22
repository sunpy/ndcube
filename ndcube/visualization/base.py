import abc
import copy


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
        when users access `ndcube.NDCube.plot`. It should strive to provide a good
        overview of the cube by default but the behaviour is left to the
        implementation.

        The ``plot()`` method **should** accept ``**kwargs``.
        """

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        memo[id(self)] = new
        for key, value in vars(self).items():
            if key != "_ndcube":
                setattr(new, key, copy.deepcopy(value, memo))
        # Reuse the parent, or its existing copy when copying an entire cube.
        new._ndcube = memo.get(id(self._ndcube), self._ndcube)
        return new
