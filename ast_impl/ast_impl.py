"""
Implementing add / delete / edit actions from LLM via AST manipulations.
"""

from copy import deepcopy
from typing import Iterable

import ast_comments as ast  # built-in `ast` module ignores comments


def _get_nodes_at_line(tree: ast.AST, line_number: int) -> Iterable[ast.AST]:
    # NOTE: does `ast.walk` always give the top node first (for specific line)?
    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        if node.lineno == line_number:
            yield node


def _get_last_node_before_line(tree: ast.AST, line_number: int) -> ast.AST | None:
    body = tree.body
    for idx in range(len(body)):
        if not hasattr(body[idx], "lineno"):
            continue
        if body[idx].lineno >= line_number:
            if idx > 0:
                return body[idx - 1]
            else:
                return body[0]
    return body[0]


def _get_parent_node(tree: ast.AST, target_node: ast.AST) -> ast.AST | None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if child == target_node:
                return node

    return None


def add_node_at_line(tree_orig: ast.AST, tree_new: ast.AST, line_number: int, before: bool = True) -> ast.AST:
    tree_orig = deepcopy(tree_orig)  # don't change the original tree

    try:
        target = next(_get_nodes_at_line(tree_orig, line_number))
        parent = _get_parent_node(tree_orig, target)
    except StopIteration:
        # no code at required line, we add it manually in-between
        target = _get_last_node_before_line(tree_orig, line_number)
        parent = tree_orig

    idx = parent.body.index(target)
    if not before:
        idx += 1
    parent.body.insert(idx, tree_new)

    return ast.fix_missing_locations(tree_orig)


def delete_node_at_line(tree: ast.AST, line_number: int) -> ast.AST:
    class NodeRemover(ast.NodeTransformer):
        def __init__(self, line_number):
            self.line_number = line_number

        def visit(self, node):
            if not hasattr(node, "lineno") or node.lineno != self.line_number:
                return self.generic_visit(node)

            # remove node at `line_number`
            return None

    res = NodeRemover(line_number).visit(deepcopy(tree))

    return ast.fix_missing_locations(res)


def edit_node_at_line(tree_orig: ast.AST, tree_new: ast.AST, line_number: int) -> ast.AST:
    """
    Edit is more like a swap: replace node at given line number with the new one.
    """
    t = deepcopy(tree_orig)  # working copy: don't change the original tree

    # find what node resides at given line number, and replace it with the new node
    target = next(_get_nodes_at_line(t, line_number))
    parent = _get_parent_node(t, target)

    # NOTE: not all nodes have a `body` attribute
    idx = parent.body.index(target)
    parent.body[idx] = tree_new

    return ast.fix_missing_locations(t)
