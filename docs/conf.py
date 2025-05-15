#
# Configuration file for the Sphinx documentation builder.
import os
import warnings
import datetime
from pathlib import Path

from astropy.utils.exceptions import AstropyDeprecationWarning
from matplotlib import MatplotlibDeprecationWarning
from packaging.version import Version

# -- Read the Docs Specific Configuration ------------------------------------
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    os.environ['HIDE_PARFIVE_PROGESS'] = 'True'

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
from ndcube import __version__

_version = Version(__version__)
version = release = str(_version)
# Avoid "post" appearing in version string in rendered docs
if _version.is_postrelease:
    version = release = _version.base_version
# Avoid long githashes in rendered Sphinx docs
elif _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"
is_development = _version.is_devrelease
is_release = not (_version.is_prerelease or _version.is_devrelease)

project = "ndcube"
author = "The SunPy Community"
copyright = f'{datetime.datetime.now().year}, {author}'  # noqa: A001

warnings.filterwarnings("error", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("error", category=AstropyDeprecationWarning)

# -- General configuration ---------------------------------------------------

# Wrap large function/method signatures
maximum_signature_line_length = 80

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
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
    'ndcube.utils.sphinx.code_context',
    'sphinx_gallery.gen_gallery',
    "sphinxext.opengraph",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_changelog",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# Treat everything in single ` as a Python reference.
default_role = "py:obj"

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/',
               (None, 'http://data.astropy.org/intersphinx/python3.inv')),
    'numpy': ('https://docs.scipy.org/doc/numpy/',
              (None, 'http://data.astropy.org/intersphinx/numpy.inv')),
    'matplotlib': ('https://matplotlib.org/',
                   (None, 'http://data.astropy.org/intersphinx/matplotlib.inv')),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'sunpy': ('https://docs.sunpy.org/en/stable/', None),
    'mpl_animators': ('https://docs.sunpy.org/projects/mpl-animators/en/stable/', None),
    'gwcs': ('https://gwcs.readthedocs.io/en/stable/', None),
    'reproject': ("https://reproject.readthedocs.io/en/stable/", None)
    }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sunpy"

html_logo = png_icon = 'logo/ndcube.png'

html_favicon = 'logo/favicon.png'

# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# By default, when rendering docstrings for classes, sphinx.ext.autodoc will
# make docs with the class-level docstring and the class-method docstrings,
# but not the __init__ docstring, which often contains the parameters to
# class constructors across the scientific Python ecosystem. The option below
# will append the __init__ docstring to the class-level docstring when rendering
# the docs. For more options, see:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"

# -- Other options ----------------------------------------------------------

napoleon_use_rtype = False
napoleon_google_docstring = False
napoleon_use_param = False

nitpicky = True
# This is not used. See docs/nitpick-exceptions file for the actual listing.
nitpick_ignore = []
for line in open('nitpick-exceptions'):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))

# -- Sphinx Gallery ---------------------------------------------------------

sphinx_gallery_conf = {
    'backreferences_dir': Path('generated/modules'),
    'filename_pattern': '^((?!skip_).)*$',
    'examples_dirs': Path('../examples'),
    'within_subsection_order': "ExampleTitleSortKey",
    'gallery_dirs': Path('generated/gallery'),
    'matplotlib_animations': True,
    "default_thumb_file": png_icon,
    'abort_on_example_error': False,
    'plot_gallery': 'True',
    'remove_config_comments': True,
    'doc_module': ('ndcube'),
    'only_warn_on_example_error': True,
}

# -- Sphinxext Opengraph ----------------------------------------------------

ogp_image = "https://github.com/sunpy/ndcube/raw/main/docs/logo/ndcube.png"
ogp_use_first_image = True
ogp_description_length = 160
ogp_custom_meta_tags = [
    '<meta property="og:ignore_canonical" content="true" />',
]
