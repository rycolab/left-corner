# An Exploration of Left-Corner Transformations

<img style="width: 40%" src="data/dall-e.jpeg" align="right">

**TL;DR** The repository implements the generalized left-corner transformation
from [our EMNLP 2023 paper](https://arxiv.org/pdf/2311.16258.pdf) and many other utilities for working with
context-free grammars.

**Abstract**:
The [left-corner transformation](https://ieeexplore.ieee.org/document/4569645/)
is used to remove left recursion from context-free grammars, which is an
important step towards making the grammar parsable top-down with simple
techniques.  This paper generalizes prior left-corner transformations to support
semiring-weighted production rules and to provide finer-grained control over
which left corners may be moved.  Our generalized left-corner transformation
(GLCT) arose from unifying the left-corner transformation and [speculation
transformation](https://www.cs.jhu.edu/~jason/papers/eisner+blatz.fg06.pdf),
originally for logic programming.  Our new transformation and speculation define
equivalent weighted languages. Yet, their derivation trees are structurally
different in an important way: GLCT replaces left recursion with right
recursion, and speculation does not.  We also provide several technical results
regarding the formal relationships between the outputs of GLCT, speculation, and
the original grammar.  Lastly, we empirically investigate the efficiency of GLCT
for left-recursion elimination from grammars of nine languages.

## Getting Started 🚦

### Installation 🔧

```bash
$ git clone git@github.com:rycolab/left-corner.git ./left-corner
$ cd left-corner
$ pip install -e .
```

### Tutorial 🎓

Please see the tutorial notebook ([Tutorial.ipynb](https://github.com/rycolab/left-corner/blob/main/Tutorial.ipynb)).

### Tests 🙋

Unit and integration tests
```bash
$ pytest tests/tests.py 
```

Code coverage
```bash
$ ./run-coverage.bash
```

### Experiments 🧪

The experiments on the English ATIS grammar can be run without downloading additional data. To run the experiments on the non-English, SPMRL grammars, you will need to acquire the SPRML treebanks from the [shared task website](https://www.spmrl.org/).
```bash
$ python test/test_grammars.py
```

## Citation 📜

If you use this repository in your work, please link to it or cite it with the following BibTeX:
```
@inproceedings{opedal2023lct,
  title = {An Exploration of Left-Corner Transformations}
  author = {Andreas Opedal and Eleftheria Tsipidi and Tiago Pimentel
            and Ryan Cotterell and Tim Vieira},
  booktitle = {Proceedings of the Conference on Empirical Methods in
               Natural Language Processing},
  year = {2023},
  url = "",
}
```
