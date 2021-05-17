#
# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'ndcube'
copyright = '2021, The SunPy Community'
author = 'The SunPy Community'

# The full version, including alpha/beta/rc tags
from ndcube import __version__

release = __version__
is_development = '.dev' in __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'ndcube.utils.sphinx.code_context',
    'sphinx_changelog',
    'pytest_doctestplus.sphinx.doctestplus',
]

# Define code to be run by doctest in front of every code block.

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = 'obj'

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/',
               (None, 'http://data.astropy.org/intersphinx/python3.inv')),
    'numpy': ('https://docs.scipy.org/doc/numpy/',
              (None, 'http://data.astropy.org/intersphinx/numpy.inv')),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/',
              (None, 'http://data.astropy.org/intersphinx/scipy.inv')),
    'matplotlib': ('https://matplotlib.org/',
                   (None, 'http://data.astropy.org/intersphinx/matplotlib.inv')),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'sunpy': ('https://docs.sunpy.org/en/stable/', None),
    'gwcs': ('https://gwcs.readthedocs.io/en/stable/', None)}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

try:
    from sunpy_sphinx_theme.conf import *
except ImportError:
    html_theme = 'default'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    '-Nfontsize=10',
    '-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Efontsize=10',
    '-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Gfontsize=10',
    '-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif'
]
