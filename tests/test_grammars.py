"""
Run experiments on natural language grammars.
"""
import re, os, argparse, dill, nltk
from datetime import timedelta
from collections import defaultdict
from tabulate import tabulate
from time import time
from path import Path

from leftcorner import Boolean, Real, CFG
from leftcorner.misc import load_atis, timeit, colors

data_dir = Path('./data/')
spmrl_dir = data_dir / 'spmrl'
outputs = Path('./output/')
out_spmrl = outputs / 'spmrl'

langs = [
    'Basque',
    'English',
    'French',
    'German',
    'Hebrew',
    'Hungarian',
    'Korean',
    'Polish',
    'Swedish',
]


class _Sym:
    def __init__(self, sym):
        self.sym = sym
        self._hash = hash(sym)
    def __repr__(self):      return str(self.sym)
    def __hash__(self):      return self._hash
    def __eq__(self, other): return isinstance(other, _Sym) and self.sym == other.sym


def cache_data(obj, fname):
    with open(fname, 'wb') as f:
        dill.dump(obj, f)


def load_data(fname):
    obj = None
    with open(fname, 'rb') as f:
        obj = dill.load(f)
    return obj


def get_spmrl_cfg(grammar_path, R, language):
    from leftcorner.cfg import CFG

    # It looks like the nonterminal and terminal alphabets might overlap in SPMRL
    def NT(X): return X
    terminals = set()
    def Sym(X):
        X = _Sym(X)
        terminals.add(X)
        return X

    def _get_start_sym(line):
        # get first symbol, remove left parenthesis
        return line.split(' ')[0][1:]

    # preprocess the treebank
    with open(grammar_path, "r") as f:
        grammar_str = f.read().strip()

    # drop the fine-graind nonterminal annotations
    grammar_str = re.sub('##.+?##', '', grammar_str)
    lines = grammar_str.split('\n')
    if grammar_str.startswith('( ('):
        lines = [l[2:-1] for l in lines]

    headcounts = defaultdict(lambda: 0)
    rulecounts = defaultdict(lambda: dict(count=0))
    # Extract rules and their counts
    for line in lines:
        if language == 'German' or language == 'swedish':
            headcounts['TOP'] += 1
            start_rule = nltk.Production(nltk.Nonterminal('TOP'), [nltk.Nonterminal(_get_start_sym(line))])
            srule_str = repr(start_rule)
            rulecounts[srule_str]['count'] += 1
            rulecounts[srule_str]['prod'] = start_rule
        productions = nltk.Tree.fromstring(line).productions()
        i = 0
        while i < len(productions):
            production = productions[i]
            lhs = production.lhs()
            headcounts[repr(lhs)] += 1
            # prevent unary cycles
            while len(production.rhs())==1 and production.is_nonlexical():
                i += 1
                production = productions[i]
            rule = nltk.Production(lhs, production.rhs())
            rule_str = repr(rule)
            rulecounts[rule_str]['count'] += 1
            rulecounts[rule_str]['prod'] = rule
            i += 1
    # Handle German and swedish SPMRL treebanks having more than one start symbol
    if language == 'German' or language == 'swedish':
        start_sym = NT('TOP')
    else:
        start_sym = NT(_get_start_sym(lines[0]))
    cfg = CFG(R=R, S=start_sym, V=terminals)
    for rule_str in rulecounts:
        production = rulecounts[rule_str]['prod']
        count = rulecounts[rule_str]['count']
        lhs = repr(production.lhs())
        headcount = headcounts[lhs]
        weight = count / headcount
        head = NT(lhs)
        rhs = production.rhs()
        tail = []
        for t in rhs:
            if isinstance(t, nltk.Nonterminal):
                tail.append(NT(repr(t)))
            else:
                tail.append(Sym(t))
        cfg.add(R(float(weight)), head, *tail)
    return cfg


def save(self, filename):
    with open(filename, 'w') as f:
        for (head, body), w in self.P:
            f.write(f'{head} → {" ".join(str(x) for x in body)} : {w.score}\n')

#@classmethod
#def load(cls, filename, semiring):
#    with open(filename, 'r') as f:
#        return cls.from_string(f.read(), semiring)


def run_experiment(language):

    print(colors.light.yellow % colors.thick_line(80))
    print(colors.light.yellow % language)

    start = time()
    results = dict()

    with timeit('Load grammar'):
        if language == 'English':
            cfg = load_atis(data_dir / 'atis-grammar.txt')
        else:
            lang_path = 'swedish' if language == 'Swedish' else language
            grammar_path = spmrl_dir / f'train5k.{lang_path}.gold.ptb'
            cfg = get_spmrl_cfg(grammar_path, Real, language)

    print('rules/size/nts/vocab:', cfg.num_rules, cfg.size, len(cfg.N), len(cfg.V))

    with timeit('scc analysis'):
        Ps = cfg.find_lr_rules()
        Xs = cfg.sufficient_Xs(Ps)

    print()

    fast_mode = False   # if True, run with filtering enabled.

    print('Raw Transformations')
    print(colors.line(80))

    with timeit('selective left-corner transformation (SLCT)'):
        before = time()
        slct = cfg.lc_selective(Ps, filter=fast_mode)
        results['time'] = {'slct': time()-before}

    with timeit('generalized left-corner transformation (GLCT)'):
        before = time()
        glct = cfg.lc_generalized(Xs, Ps, filter=fast_mode)
        results['time'].update({'glct': time()-before})
    print(colors.thick_line(80))

    print('Trim')
    print(colors.line(80))
    with timeit('SLCT + Trim'):
        before = time()
        t_slct = slct.trim()
        results['time'].update({'t_slct': time()-before})

    assert not t_slct.is_left_recursive()

    with timeit('GLCT + Trim'):
        before = time()
        t_glct = glct.trim()
        results['time'].update({'t_glct': time()-before})
    assert not t_glct.is_left_recursive()

    print(colors.thick_line(80))
    print('Nullary Removal')
    print(colors.line(80))
    with timeit('SLCT + Trim + Nullary Removal'):
        before = time()
        e_slct = t_slct.nullaryremove().trim()
        results['time'].update({'e_slct': time()-before})
    assert not e_slct.is_left_recursive()

    with timeit('GLCT + Trim + Nullary Removal'):
        before = time()
        e_glct = t_glct.nullaryremove().trim()
        results['time'].update({'e_glct': time()-before})
    assert not e_glct.is_left_recursive()
    print(colors.line(80))

    sec = time()-start
    print("total time:", '%.4f sec' % sec)

    # Print tables
    headers = "Raw\tTrim\tTrim + ε-removal".split('\t')

    selective = [slct, t_slct, e_slct]
    generalized = [glct, t_glct, e_glct]

    print(colors.thick_line(80))
    print("Grammar size")
    print(colors.line(80))
    data = [["Original grammar", f'{cfg.size:,}'] + ['', ''],
    ["Selective"]+[f'{g.size:,}' for g in selective],
    ["Generalized"]+[f'{g.size:,}' for g in generalized]]
    print(tabulate(data, headers=headers, stralign="right"))
    print()

    print("Number of rules")
    print(colors.line(80))
    data = [["Original grammar", f'{cfg.num_rules:,}'] + ['', ''],
    ["Selective"]+[f'{g.num_rules:,}' for g in selective],
    ["Generalized"]+[f'{g.num_rules:,}' for g in generalized]]
    print(tabulate(data, headers=headers, stralign="right"))
    print()

    # save grammars and results
    if 1:
        print(colors.thick_line(80))
        print('Storing the outputs')

        output_dir = outputs / 'atis' if language == 'English' else out_spmrl / language
        pt_dir = output_dir / 'plaintext'
        pickle_dir = output_dir / 'pickles'
        output_dir.makedirs_p()
        pt_dir.makedirs_p()
        pickle_dir.makedirs_p()

        gr_names = 'cfg slct t_slct e_slct glct t_glct e_glct'.split()
        grammars = [cfg] + selective + generalized

        with timeit('save results'):
            results['size'] = {gr_names[i]: grammars[i].size for i in range(len(gr_names))}
            results['num_rules'] = {gr_names[i]: grammars[i].num_rules for i in range(len(gr_names))}
            cache_data(results, output_dir / 'results.pkl')

        # save grammars (kind of slow)
        if 0:
            with timeit('save scc analysis'):
                cache_data(Ps, output_dir / 'Ps.pkl')
                cache_data(Xs, output_dir / 'Xs.pkl')

            gr_name_dict = dict(zip(gr_names[1:], grammars[1:]))

            for gr_name, grammar in gr_name_dict.items():
                with timeit(gr_name):
                    save(grammar, pt_dir / f'{gr_name}.txt')
                    # don't pickle raw grammars
                    if gr_name == 'slct' or gr_name == 'glct': continue
                    cache_data(grammar, pickle_dir/ f'{gr_name}.pkl')

    print(colors.thick_line(80))
    print("LR removed in SLCT + Trim?", colors.mark(not t_slct.is_left_recursive()))
    print("LR removed in SLCT + Trim + Nullary Removal?", colors.mark(not e_slct.is_left_recursive()))
    print(colors.line(80))

    print("LR removed in GLCT + Trim?", colors.mark(not t_glct.is_left_recursive()))
    print("LR removed in GLCT + Trim + Nullary Removal?", colors.mark(not e_glct.is_left_recursive()))
    print(colors.line(80))

    print(colors.light.yellow % "Done")
    print(colors.light.yellow % colors.thick_line(80))
    print()


def main():
    parser = argparse.ArgumentParser(description="Run experiments for SPMRL grammars and ATIS")
    parser.add_argument('--langs', nargs='*', choices=langs, default=None)
    args = parser.parse_args()
    languages = args.langs if args.langs else langs
    for lang in languages:
        run_experiment(lang)


if __name__ == '__main__':
    main()
