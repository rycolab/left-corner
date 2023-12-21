from setuptools import setup

setup(
    name='leftcorner',
    version='1.0.1',
    description=(
        "Implementation of the generalized left-corner transformation (Opedal et al. 2023), "
        "an efficient method for eliminating left recursion from context-free grammars, "
        "and many other utilities for working with context-free grammars."
    ),
    project_url = 'https://github.com/rycolab/left-corner',
    install_requires = [
        'numpy',
        'IPython',
        'networkx',
        'nltk',
        'svgling',    # nltk uses svgling to draw derivations
        'tabulate',
        'dill',
        'pytest',
        'graphviz',   # for notebook visualization of left-recursion graph
        'path'
    ],
    authors = [
        'Andreas Opedal',
        'Eleftheria Tsipidi'
        'Tiago Pimentel',
        'Ryan Cotterell',
        'Tim Vieira',
    ],
    readme=open('README.md').read(),
    scripts=[],
    packages=['leftcorner'],
)
