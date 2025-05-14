import ast_comments as ast

from ast_impl import add_node_at_line, delete_node_at_line, edit_node_at_line


def test_add_simple():
    orig_tree = ast.parse(
        """
import os
os.system('ls')
    """.strip()
    )
    new_tree = ast.parse("xs = [1,2,3]")

    res = add_node_at_line(orig_tree, new_tree, line_number=2, before=True)
    assert ast.unparse(res) == ast.unparse(
        ast.parse(
            """
import os
xs = [1,2,3]
os.system('ls')
""".strip()
        )
    )

    res = add_node_at_line(orig_tree, new_tree, line_number=2, before=False)
    assert ast.unparse(res) == ast.unparse(
        ast.parse(
            """
import os
os.system('ls')
xs = [1,2,3]
""".strip()
        )
    )


def test_add_in_ifelse():
    orig_tree = ast.parse(
        """
if True:
    print("true")
else:
    print("false")
""".strip()
    )
    new_tree = ast.parse("pass")

    res = add_node_at_line(orig_tree, new_tree, line_number=2, before=True)
    assert ast.unparse(res) == ast.unparse(
        ast.parse(
            """
if True:
    pass
    print("true")
else:
    print("false")
""".strip()
        )
    )


def test_delete_simple():
    orig_tree = ast.parse(
        """
import os
def foo(x, y):
    return x + y
os.system('ls')
    """.strip()
    )

    res = delete_node_at_line(orig_tree, line_number=1)
    assert ast.unparse(res) == ast.unparse(
        ast.parse(
            """
def foo(x, y):
    return x + y
os.system('ls')
""".strip()
        )
    )

    res = delete_node_at_line(orig_tree, line_number=2)
    assert ast.unparse(res) == ast.unparse(
        ast.parse(
            """
import os
os.system('ls')
""".strip()
        )
    )


def test_edit_simple():
    orig_tree = ast.parse(
        """
import foo

def bar(x, y):
    return foo.max(x, y) * 2

pass
""".strip()
    )
    new_tree = ast.parse("xs = [1, 2, 3]")

    res = edit_node_at_line(orig_tree, new_tree, line_number=3)
    assert ast.unparse(res) == ast.unparse(
        ast.parse(
            """
import foo
xs = [1, 2, 3]
pass
""".strip()
        )
    )
