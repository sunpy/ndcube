#
# Configuration file for the Sphinx documentation builder.
import os
import warnings
import datetime

from astropy.utils.exceptions import AstropyDeprecationWarning
from matplotlib import MatplotlibDeprecationWarning
from packaging.version import Version

# -- Read the Docs Specific Configuration
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
    'sphinx_gallery.gen_gallery',
    "sphinxext.opengraph",
]

# -- Sphinxext Opengraph
ogp_image = "https://github.com/sunpy/ndcube/raw/main/docs/logo/ndcube.png"
ogp_use_first_image = True
ogp_description_length = 160
ogp_custom_meta_tags = [
    '<meta property="og:ignore_canonical" content="true" />',
]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'
# Treat everything in single ` as a Python reference.
default_role = "py:obj"
napoleon_use_rtype = False
napoleon_google_docstring = False
napoleon_use_param = False
# TODO: Enable this in future.
nitpicky = True
# This is not used. See docs/nitpick-exceptions file for the actual listing.
nitpick_ignore = []
for line in open('nitpick-exceptions'):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))

<<<<<<<
=======
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
>>>>>>>
# -- Options for intersphinx extension
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

# -- Options for HTML output
html_theme = "sunpy"
html_logo = png_icon = 'logo/ndcube.png'
html_favicon = 'logo/favicon.png'
graphviz_output_format = 'svg'
graphviz_dot_args = [
    '-Nfontsize=10',
    '-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Efontsize=10',
    '-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Gfontsize=10',
    '-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif'
]


# -- Sphinx Gallery
sphinx_gallery_conf = {
    'backreferences_dir': os.path.join('generated', 'modules'),
    'filename_pattern': '^((?!skip_).)*$',
    'examples_dirs': os.path.join('..', 'examples'),
    'within_subsection_order': "ExampleTitleSortKey",
    'gallery_dirs': os.path.join('generated', 'gallery'),
    'matplotlib_animations': True,
    "default_thumb_file": png_icon,
    'abort_on_example_error': False,
    'plot_gallery': 'True',
    'remove_config_comments': True,
    'doc_module': ('ndcube'),
    'only_warn_on_example_error': True,
}
