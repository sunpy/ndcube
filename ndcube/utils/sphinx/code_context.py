"""
A sphinx directive to execute a code block with context, and to provide the
user a way to see that context.
"""

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.directives.code import CodeBlock


class ExpandingCodeBlock(CodeBlock):
    """
    A code-block directive which is wrapped in a HTML details tag.

    It behaves like the code-block directive, with the addition of a
    ``:summary:`` option, which sets the unexpanded text.
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = False
    option_spec = {
        'force': directives.flag,
        'linenos': directives.flag,
        'dedent': int,
        'lineno-start': int,
        'emphasize-lines': directives.unchanged_required,
        'caption': directives.unchanged_required,
        'class': directives.class_option,
        'name': directives.unchanged,
        'summary': directives.unchanged_required,
    }

    def run(self):
        source, lineno = self.state_machine.get_source_and_line(self.lineno)

        summary_text = self.options.get('summary', 'Show setup code')

        opening_details = f"""\
        <details>
          <summary style="display: list-item; cursor: pointer;">{summary_text}</summary>
        """
        open_raw_node = nodes.raw('', opening_details, format='html')
        open_raw_node.source, open_raw_node.line = source, lineno

        close_raw_node = nodes.raw('', "</details>", format='html')
        close_raw_node.source, close_raw_node.line = source, lineno

        literal = super().run()[0]

        return [open_raw_node, literal, close_raw_node]


def setup(app):
    app.add_directive('expanding-code-block', ExpandingCodeBlock)

    return {'parallel_read_safe': True, 'parallel_write_safe': True}
