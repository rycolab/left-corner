import re
import dill
import sys
import networkx as nx
import nltk
import numpy as np
from path import Path
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain, combinations
from time import time
from IPython.display import display, SVG, Image, HTML, Latex


def format_table(rows, headings=None):
    def fmt(x):
        try:
            return x._repr_html_()
        except AttributeError:
            try:
                return x._repr_svg_()
            except AttributeError:
                return str(x)

    return (
        '<table>'
         + ('<tr style="font-weight: bold;">' + ''.join(f'<td>{x}</td>' for x in headings) +'</tr>' if headings else '')
         + ''.join(f'<tr>' + ''.join(f'<td>{fmt(x)}</td>' for x in row) +  ' </tr>' for row in rows)
         + '</table>'
    )


def display_table(*args, **kwargs):
    return display(HTML(format_table(*args, **kwargs)))


def assert_equal_chart(have, want, domain=None, tol=1e-5, verbose=False, throw=True):
    if domain is None: domain = have.keys() | want.keys()
    assert verbose or throw
    for x in domain:
        if have[x].metric(want[x]) <= tol:
            if verbose:
                print(colors.mark(True), x, have[x])
        else:
            if verbose:
                print(colors.mark(False), x, have[x], want[x])
            if throw:
                raise AssertionError(f'{x}: {have[x]} {want[x]}')


@contextmanager
def timeit(name, fmt='{name} ({htime})', header=None):
    """Context Manager which prints the time it took to run code block."""
    if header is not None: print(header)
    b4 = time()
    yield
    sec = time() - b4
    ht = '%.4f sec' % sec
    print(fmt.format(name=name, htime=ht, sec=sec), file=sys.stderr)


def load_atis(path):
    """
    Load the ATIS grammar which was used in Moore's paper on left-recursion elimination.
    See https://users.sussex.ac.uk/~johnca/cfg-resources/index.html
    """
    from leftcorner.cfg import CFG, Boolean
    with open(path, 'r') as f:
        content = f.read().strip()
    content = content.replace(":", "-colon-").replace("non_cyclic", "NON_CYCLIC")
    content = content.replace("``", "-quot_ini-").replace("''", "-quot_fin-")
    lines = content.split('\n')
    assert lines[0][0] != ";"
    #if moore_grammar[0][0] == ";":
    #    # clean ct-grammar-eval from initial comments
    #    moore_grammar = moore_grammar[5:]
    rules = []
    prev, lhs = None, None
    for line in lines:
        line = line.strip()
        if prev == None or prev == "":
            lhs = "S" if line == "SIGMA" else line
        elif line != "" and lhs != line:
            rules.append(f"{True}: {lhs} → {line}")
        prev = line
    return CFG.from_string("\n".join(rules), Boolean)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def ansi(color=None, light=None, bg=3):
    return '\x1b[%s;%s%sm' % (light, bg, color) + '%s\x1b[0m'


class colors:

    black, red, green, yellow, blue, magenta, cyan, white = \
        [ansi(c, 0) for c in range(8)]

    class light:
        black, red, green, yellow, blue, magenta, cyan, white = \
            [ansi(c, 1) for c in range(8)]

    class dark:
        black, red, green, yellow, blue, magenta, cyan, white = \
            [ansi(c, 2) for c in range(8)]

    def rgb(r,g,b): return f"\x1b[38;2;{r};{g};{b}m%s\x1b[0m"

    orange = rgb(255, 165, 0)

    purple = '\x1b[38;5;91m' + '%s' + '\x1b[0m'

    normal = '\x1b[0m%s\x1b[0m'
    bold = '\x1b[1m%s\x1b[0m'
    italic = "\x1b[3m%s\x1b[0m"
    underline = "\x1b[4m%s\x1b[0m"
    strike = "\x1b[9m%s\x1b[0m"
    #overline = lambda x: (u''.join(unicode(c) + u'\u0305' for c in unicode(x))).encode('utf-8')

    def line(n): return '─'*(n)

    def thick_line(n): return ('━'*n)

    check = green % '✔'
    xmark = dark.red % '✘'
    def mark(x): return colors.check if x else colors.xmark
