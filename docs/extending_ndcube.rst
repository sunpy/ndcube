.. _extending_ndcube:

**************************************
Extending ``ndcube`` in Other Packages
**************************************

This section is aimed at people developing packages which extend the functionality of ``ndcube`` data classes.

Requiring ``ndcube`` in your package
====================================

ndcube has required dependencies on ``astropy``, ``gwcs`` and `numpy`.
In addition to this it has two sets of optional dependencies:

* ``plotting``, which requires `matplotlib` and `mpl_animators`.
  If you are using any of the ``ndcube`` objects directly with the default plotting implementation you probably want to include this extra in your requirements.
* ``reproject``, which requires the `reproject` package.
  This is required to use the :meth:`ndcube.NDCube.reproject_to` method.

When including ndcube in your package requirements you should include either of these extras using the ``ndcube[plotting]`` type syntax.
If you wish for all extra requirements to be installed you can use ``ndcube[all]``.

Subclassing an ``ndcube`` Data Class
====================================

Before you subclass
-------------------

The data classes in ndcube have been designed to be subclassed and extended.
One word of warning before you write ``myclass(NDCube)`` though: think about if the functionality you are writing is more generally applicable.
Imagine for a moment that you are writing a class which is specifically for a cube with space, space, wavelength axes so that you can do some operations like fitting along the wavelength axis.
Your subclass of `~ndcube.NDCube` may provide great functionality for the users who have data which meet your assumptions.
However, if I come along with a space, space, wavelength, time 4D cube I may want to make use of your fitting functions along the wavelength axis.

**Can you write your functionality in a way that takes any ndcube object with any specific combination of physical types?**

Subclassing
-----------

When you are subclassing, try to use as much of the upstream object properties as possible.
Doing this will enable you to reuse functionality designed for all ndcube classes and hence make your life easier.
For instance if your class has a special property ``.info`` it would not automatically be carried through operations such as slicing and reprojecting.
You would need to customize all these operations.
If instead you put an ``info`` key in the ``meta`` dictionary it would automatically be copied through the appropriate operations.

If your subclass does have custom attributes you need to propagate through methods and functions, you will in most cases need to overload these methods in your subclass.
On `~ndcube.NDCube` the only method which returns another instance of your subclass is currently `ndcube.NDCube.reproject_to`.

.. _customizing_plotter:

Customizing the visualization
-----------------------------

The ``.plotter`` attribute of the `~ndcube.NDCube` class is configurable and allows you to customize the visualization functionality.
This can be as minor as changing the defaults or as complete as using an alternative visualization library.
To customize the default type for your subclass, you first need to implement a custom class inheriting from `ndcube.visualization.BasePlotter`.
This class should implement a default ``.plot`` method which will be called by the :meth:`ndcube.NDCube.plot` method.
As many other methods as you wish can be implemented on the plotter, to be called using the ``.plotter.mymethod()`` API.

Once you have a custom plotter class (e.g. ``CustomPlotter``) you can set this to be the default for all instances of your subclass by doing the following

.. code-block:: python

  from ndcube.visualization import BasePlotter, PlotterDescriptor
  from ndcube import NDCube

  class CustomPlotter(BasePlotter):
      def plot(self):
          pass

  class CustomCube(NDCube):
      plotter = PlotterDescriptor(default_type=CustomPlotter)
