Extending ndcube in other packages
==================================

This section of the documentation is aimed at people developing packages which extend the functionality in ndcube in other packages.


Subclassing an ndcube object
----------------------------


Before you subclass
###################

The data classes in ndcube have been designed to be subclassed and extended.
One word of warning before you go ahead and write ``myclass(NDCube)``: think about if the functionality you are writing is more generally applicable.
Imagine for a moment that you are writing a class which is specifically for a cube with space, space, wavelength axes so that you can do some operations like fitting along the wavelength axis.
Your subclass of ``NDCube`` may provide great functionality for the users who have data which meets your assumptions.
However, if I come along with a space, space, wavelength, time 4D cube I may want to make use of your fitting functions along the wavelength axis.

**Can you write your functionality in a way which takes any ``NDCube`` object which has any specific combination of physical types?**


Subclassing
###########

When you are subclassing, try to use as much of the upstream object properties as possible.
Doing this will make your life easier, by being able to reuse functionality designed for all different types of ndcube classes.
For instance if your class has a special property ``.info`` it would not automatically be carried through operations such as slicing and reprojecting, you would need to customise all these operations.
If instead you put an ``info`` key in the ``meta`` dictionary it would automatically be copied through the appropriate operations.


Customising the visualisation
#############################

The ``.plotter`` attribute of the ``NDCube`` class is configurable and allows you to customise the visualization functionality.
This can be as minor as changing the defaults or as complete as using an alternative visualization library.
To customise the default type for your subclass first you need to implement a custom class inheriting from `ndcube.visualization.BasePlotter`.
This class should implement a default ``.plot`` method which will be called by the ``NDCube.plot`` method.
As many other methods as you wish can be implemented on the plotter, to be called using the ``.plotter.mymethod()`` API.

Once you have a custom plotter class (e.g. ``CustomPlotter``) you can set this to be the default for all instances of your subclass by doing the following::

  from ndcube.visualization import BasePlotter, PlotterDescriptor
  from ndcube import NDCube

  class CustomPlotter(BasePlotter):
      def plot(self):
          pass

  class CustomCube(NDCube):
      plotter = PlotterDescriptor(default_type=CustomPlotter)
