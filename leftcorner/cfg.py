import re
import nltk
import numpy as np
import networkx as nx
import graphviz

from collections import defaultdict, Counter
from functools import cached_property
from itertools import product

from leftcorner.semiring import Semiring, Boolean
from leftcorner.misc import colors


def _gen_nt():
    _gen_nt.i += 1
    return f'@{_gen_nt.i}'
_gen_nt.i = 0


class Slash:

    def __init__(self, Y, Z):
        self.Y, self.Z = Y, Z
        self._hash = hash((Y, Z))

    def __repr__(self):
        return f'{self.Y}/{self.Z}'

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Slash) \
                and self.Y == other.Y \
                and self.Z == other.Z


class Frozen:

    def __init__(self, X):
        self._hash = hash(X)
        self.X = X

    def __repr__(self):
        return f"~{self.X}"

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Frozen) and self.X == other.X


class Rule:

    def __init__(self, w, head, body):
        self.w = w
        self.head = head
        self.body = body
        self._hash = hash((head, body))

    def __iter__(self):
        return iter((self.head, self.body))

    def __eq__(self, other):
        return (isinstance(other, Rule)
                and self.w == other.w
                and self._hash == other._hash
                and other.head == self.head
                and other.body == self.body)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f'{self.w}: {self.head} → {" ".join(map(str, self.body))}'


class Derivation:

    def __init__(self, r, x, *ys):
        assert isinstance(r, Rule) or r is None
        self.r = r
        self.x = x
        self.ys = ys

    # Warning: Currently, Derivations compare equal even if they have different rules.
    def __hash__(self):
#        return hash((self.r, self.x, self.ys))
        return hash((self.x, self.ys))

    def __eq__(self, other):
#        return (self.r, self.x, self.ys) == (other.r, other.x, other.ys)
        return isinstance(other, Derivation) and (self.x, self.ys) == (other.x, other.ys)

    def __repr__(self):
        open = colors.dark.white % '('
        close = colors.dark.white % ')'
        children = ' '.join(str(y) for y in self.ys)
        return f'{open}{self.x} {children}{close}'

    def weight(self):
        "Compute this weight this `Derivation`."
        W = self.r.w
        for y in self.ys:
            if isinstance(y, Derivation):
                W *= y.weight()
        return W

    def Yield(self):
        if isinstance(self, Derivation):
            return [w for y in self.ys for w in Derivation.Yield(y)]
        else:
            return [self]

    def to_nltk(self):
        if not isinstance(self, Derivation): return self
        return nltk.Tree(str(self.x), [Derivation.to_nltk(y) for y in self.ys])

    def _repr_html_(self):
        return self.to_nltk()._repr_svg_()


class CFG:

    def __init__(self, R: 'semiring', S: 'start symbol', V: 'terminal vocabulary'):

        # semiring
        self.R = R

        # alphabet
        self.V = V

        # nonterminals
        self.N = {S}

        # rules
        self.rules = []

        # unique start symbol
        self.S = S

    def __repr__(self):
        return "\n".join(f"{p}" for p in self)

    def _repr_html_(self):
        return f'<pre style="width: fit-content; text-align: left; border: thin solid black; padding: 0.5em;">{self}</pre>'

    @classmethod
    def from_string(cls, string, semiring, comment="#", start='S'):
        V = set()
        cfg = cls(R=semiring, S=start, V=V)
        string = string.replace('->', '→')   # synonym for the arrow
        for line in string.split('\n'):
            line = line.strip()
            if not line or line.startswith(comment): continue
            try:
                [(w, lhs, rhs)] = re.findall('(.*):\s*(\S+)\s*→\s*(.*)$', line)
                lhs = lhs.strip()
                rhs = rhs.strip().split()

                for x in rhs:
                    if not x[0].isupper():
                        V.add(x)

                cfg.add(semiring.from_string(w), lhs, *rhs)

            except ValueError as e:
                raise ValueError(f'bad input line:\n{line}')
        return cfg

    @cached_property
    def rhs(self):
        rhs = defaultdict(list)
        for r in self:
            rhs[r.head].append(r)
        return rhs

    def is_terminal(self, x):
        return x in self.V

    def is_nonterminal(self, X):
        return not self.is_terminal(X)

    def __iter__(self):
        return iter(self.rules)

    @property
    def size(self):
        return sum(1 + len(r.body) for r in self)

    @property
    def num_rules(self):
        return len(self.rules)

    def spawn(self, *, R=None, S=None, V=None):
        return CFG(R=self.R if R is None else R,
                   S=self.S if S is None else S,
                   V=set(self.V) if V is None else V)

    def add(self, w, head, *body):
        if w == self.R.zero: return   # skip rules with weight zero
        assert isinstance(w, Semiring), w
        self.N.add(head)
        self.rules.append(Rule(w, head, body))

    def assert_equal(self, other, verbose=False, throw=True):
        assert verbose or throw
        if verbose:
            # TODO: need to check the weights in the print out; we do it in the assertion
            S = set(self.rules)
            G = set(other.rules)
            for r in sorted(S | G, key=str):
                if r in S and r in G: continue
                #if r in S and r not in G: continue
                #if r not in S and r in G: continue
                print(
                    colors.mark(r in S),
                    #colors.mark(r in S and r in G),
                    colors.mark(r in G),
                    r,
                )
        assert not throw or Counter(self.rules) == Counter(other.rules), \
            f'\n\nhave=\n{str(self)}\nwant=\n{str(other)}'

    def treesum(self, **kwargs):
        return self.agenda()[self.S]

    def trim(self, bottomup_only=False):

        C = set(self.V)
        C.update(e.head for e in self.rules if len(e.body) == 0)

        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for e in self.rules:
            incoming[e.head].append(e)
            for b in e.body:
                outgoing[b].append(e)

        agenda = set(C)
        while agenda:
            x = agenda.pop()
            for e in outgoing[x]:
                if all((b in C) for b in e.body):
                    if e.head not in C:
                        C.add(e.head)
                        agenda.add(e.head)

        if bottomup_only: return self._trim(C)

        T = {self.S}
        agenda.update(T)
        while agenda:
            x = agenda.pop()
            for e in incoming[x]:
                #assert e.head in T
                for b in e.body:
                    if b not in T and b in C:
                        T.add(b)
                        agenda.add(b)

        return self._trim(T)

    def cotrim(self):
        return self.trim(bottomup_only=True)

    def _trim(self, symbols):
        new = self.spawn()
        for p in self:
            if p.head in symbols and p.w != self.R.zero and set(p.body) <= symbols:
                new.add(p.w, p.head, *p.body)
        return new

    def derivations(self, X, H):
        "Enumerate derivations of symbol X with height <= H"

        if isinstance(X, tuple):
            if len(X) == 0:
                yield ()
            else:
                for x in self.derivations(X[0], H):
                    for xs in self.derivations(X[1:], H):
                        yield (x, *xs)

        elif self.is_terminal(X):
            yield X

        elif H <= 0:
            return

        else:
            for r in self.rhs[X]:
                for ys in self.derivations(r.body, H-1):
                    yield Derivation(r, X, *ys)

    def derivations_of(self, s):
        "Enumeration of derivations with yield `s`"

        def p(X,I,K):
            if isinstance(X, tuple):
                if len(X) == 0:
                    if K-I == 0:
                        yield ()
                else:
                    for J in range(I, K+1):
                        for x in p(X[0], I, J):
                            for xs in p(X[1:], J, K):
                                yield (x, *xs)
            elif self.is_terminal(X):
                if K-I == 1 and s[I] == X:
                    yield X
                else:
                    return
            else:
                for r in self.rhs[X]:
                    for ys in p(r.body, I, K):
                        yield Derivation(r, X, *ys)

        return p(self.S, 0, len(s))

    #___________________________________________________________________________
    # Transformations

    def _lehmann(self, N, W):
        "Lehmann's (1977) algorithm."

        V = W.copy()
        U = W.copy()

        for j in N:
            V, U = U, V
            V = self.R.chart()
            s = U[j, j].star()
            for i in N:
                for k in N:
                    # i ➙ j ⇝ j ➙ k
                    V[i, k] = U[i, k] + U[i, j] * s * U[j, k]

        # add paths of length zero
        for i in N:
            V[i, i] += self.R.one

        return V

    def unaryremove(self):
        """
        Return an equivalent grammar with no unary rules.
        """

        # compute unary chain weights
        A = self.R.chart()
        for p in self.rules:
            if len(p.body) == 1 and self.is_nonterminal(p.body[0]):
                A[p.body[0], p.head] += p.w

        W = self._lehmann(self.N, A)

        new = self.spawn()
        for p in self.rules:
            X, body = p
            if len(body) == 1 and self.is_nonterminal(body[0]): continue
            for Y in self.N:
                new.add(W[X,Y]*p.w, Y, *body)

        return new

    def nullaryremove(self, binarize=True):
        """
        Return an equivalent grammar with no nullary rules except for one at the start symbol.
        """

        # A really wide rule can take a very long time because of the power set
        # in this rule so it is really important to binarize.
        if binarize: self = self.binarize()

        ecfg = self.spawn(V=set())
        for p in self:
            if not any(self.is_terminal(elem) for elem in p.body):
                ecfg.add(p.w, p.head, *p.body)

        null_weight = ecfg.agenda()

        return self._push_null_weights(null_weight)

    def _push_null_weights(self, null_weight):
        rcfg = self.spawn()
        rcfg.add(null_weight[self.S], self.S)

        for p in self:
            head, body = p

            if len(body) == 0: continue  # drop nullary rule

            for B in product([0, 1], repeat=len(body)):
                v, lst = p.w, []

                for i, b in enumerate(B):
                    if b:
                        v *= null_weight[body[i]]
                    else:
                        lst.append(body[i])

                # excludes the all zero case
                if len(lst) > 0:
                    rcfg.add(v, head, *lst)

        return rcfg

    def separate_terminals(self):
        new = self.spawn()
        for p in self.rules:
            I = [(i, i) for i, x in enumerate(p.body) if self.is_terminal(x)]
            if len(I) == 0:
                new.add(p.w, p.head, *p.body)
            else:
                for q in self._fold(p, I):
                    new.add(q.w, q.head, *q.body)
        return new

    def binarize(self):
        new = self.spawn()

        stack = list(self.rules)
        while stack:
            p = stack.pop()
            if len(p.body) <= 2:
                new.add(p.w, p.head, *p.body)
            else:
                stack.extend(self._fold(p, [(0, 1)]))

        return new

    def _fold(self, p, I):

        # new productions
        P, heads = [], []
        for (i, j) in I:
            head = _gen_nt()
            heads.append(head)
            body = p.body[i:j+1]
            P.append(Rule(self.R.one, head, body))

        # new "head" production
        body = tuple()
        start = 0
        for (end, n), head in zip(I, heads):
            body += p.body[start:end] + (head,)
            start = n+1
        body += p.body[start:]
        P.append(Rule(p.w, p.head, body))

        return P

    def cnf(self):
        return self.separate_terminals().nullaryremove().unaryremove().binarize().trim()

    def in_cnf(self):
        """check if grammar is in cnf"""
        for p in self:
            (head, body) = p
            if head == self.S and len(body) == 0:
                # S →
                continue
            elif (
                head in self.N
                and len(body) == 2
                and all([elem in self.N and elem != self.S for elem in body])
            ):
                # A → B C
                continue
            elif (
                head in self.N
                and len(body) == 1
                and body[0] in self.V
            ):
                # A → a
                continue
            else:
                return False
        return True

    def unfold(self, p, i):
        assert self.is_nonterminal(p.body[i])
        assert p in self.rules

        wp = self.R.zero
        new = self.spawn()
        for q in self:
            if q == p:
                wp += q.w
                continue
            new.add(q.w, q.head, *q.body)

        for q in self:
            if q.head == p.body[i]:
                new.add(q.w*wp, p.head, *p.body[:i], *q.body, *p.body[i+1:])

        return new

    def _slash(self, X, Y):
        # TODO: need to be a unique symbol
        return Slash(X, Y)

    def _frozen(self, X):
        # TODO: needs to be a unique symbol
        return X if self.is_terminal(X) else Frozen(X)

    def speculate(self, Xs, Ps, filter=True, id=0):
        """
        The speculation transformation as described in Opedal et al., (2023).
        """
        return Speculation(parent = self, Xs = Xs, Ps = Ps, filter = filter, id = id)

    def lc_generalized(self, Xs, Ps, filter=True, id=0):
        """
        The generalized left-corner transformation (Opedal et al., 2023)
        """
        return GLCT(parent = self, Xs = Xs, Ps = Ps, filter = filter, id = id)

    def lc_selective(self, Ps, filter=True):
        """
        The selective left-corner transformation (Johnson and Roark, 2000) with
        their top-down factoring optimization (see §2.5).
        """
        return self.lc_generalized(Ps=Ps, Xs=self.V | self.N, filter=filter)

    def agenda(self, tol=1e-12):
        "Agenda-based semi-naive evaluation"
        old = self.R.chart()

        # precompute the mapping from updates to where they need to go
        routing = defaultdict(list)
        for r in self.rules:
            for k in range(len(r.body)):
                routing[r.body[k]].append((r, k))

        change = self.R.chart()
        for a in self.V:
            change[a] += self.R.one

        for r in self.rules:
            if len(r.body) == 0:
                change[r.head] += r.w

        while len(change) > 0:
            u,v = change.popitem()

            new = old[u] + v

            if self.R.metric(old[u], new) <= tol: continue
            #if old[u] == new: continue

            for r, k in routing[u]:

                W = r.w
                for j in range(len(r.body)):
                    if u == r.body[j]:
                        if j < k:    W *= new
                        elif j == k: W *= v
                        else:        W *= old[u]
                    else:
                        W *= old[r.body[j]]

                change[r.head] += W

            old[u] = new

        return old

    def naive_bottom_up(self, tol=1e-12, timeout=100_000):

        def _approx_equal(U, V):
            return all((self.R.metric(U[X], V[X]) <= tol) for X in self.N)

        R = self.R
        V = R.chart()
        counter = 0
        while counter < timeout:
            U = self._bottom_up_step(V)
            if _approx_equal(U, V): break
            V = U
            counter += 1
        return V

    def _bottom_up_step(self, V):
        R = self.R
        one = R.one
        U = R.chart()
        for a in self.V:
            U[a] = one
        for p in self.rules:
            update = p.w
            for X in p.body:
                if self.is_nonterminal(X):
                    update *= V[X]
            U[p.head] += update
        return U

    #___________________________________________________________________________
    # Left-recursion analysis and elimination methods

    def left_recursion_graph(self):
        "Left corner graph over symbols and all rules."
        return self._left_recursion_graph(self.rules)

    def _left_recursion_graph(self, Ps):
        """
        Return the left-corner graph over all symbols given rules `Ps` In this graph,
        the nodes are the symbols (N | V) and the edges are from `body[0] →
        head` for each rule in `Ps`.  For the head to body graph, simply call
        `graph.reverse()`.
        """
        G = nx.DiGraph()
        for x in self.N | self.V:
            G.add_node(x)
        for p in Ps:
            if len(p.body) == 0: continue
            G.add_edge(p.head, p.body[0], label=p)

        # TODO: use subclassing instead of this monkey-patch workaround
        def _repr_html_():
            # add a nicer visualization for notebooks
            GG = graphviz.Digraph(
                node_attr=dict(shape='record',fontname='Monospace', fontsize='10',
                               height='0', width='0', margin="0.055,0.042"),
                edge_attr=dict(arrowhead='vee', arrowsize='0.5',
                               fontname='Monospace', fontsize='9'),
            )
            for i,j in G.edges:
                GG.edge(str(i), str(j))
            for i in G.nodes:
                GG.node(str(i))
            return GG._repr_image_svg_xml()

        # monkeypatch a nicer visualization method for notebooks
        G._repr_html_ = _repr_html_

        return G

    def is_left_recursive(self):
        "Return true iff this grammar contains any cyclical left-recursion"
        return len(self.find_lr_rules()) != 0

    def find_lr_rules(self):
        """
        Return the set of left-recursive rules (i.e., those that appear in any
        cyclical left-recursive block)
        """
        # this utility flattens the list of sets returned by `find_lr_block`
        G = self.left_recursion_graph()
        H = nx.condensation(G)
        f = H.graph['mapping']
        return {r for r in self.rules if len(r.body) > 0 and f[r.head] == f[r.body[0]]}

    def sufficient_Xs(self, Ps):
        """
        Determine the set of left corner recognition symbols required for GLCT to
        eliminate left recursion according to Theorem 4.
        """
        return ((self.V | {p.head for p in set(self.rules) - set(Ps)})
                & {p.body[0] for p in Ps})

    def elim_left_recursion(self, **kwargs):
        "Eliminate left recursion from this grammar."
        Ps = self.find_lr_rules()
        return self.lc_generalized(Xs=self.sufficient_Xs(Ps), Ps=Ps, **kwargs)


class Speculation(CFG):

    def __init__(self, parent, Xs, Ps, filter, id):
        assert set(Ps) <= set(parent.rules)
        assert all(len(r.body) > 0 for r in Ps)

        super().__init__(R=parent.R, S=parent.S, V=set(parent.V))

        self.Xs = Xs
        self.Ps = Ps
        self.filter = filter
        self.id = id
        self.parent = parent

        slash = self._slash; frozen = self._frozen; one = self.R.one
        add = self.add

        # slash base case
        for X in (Xs if filter else (parent.V | parent.N)):
            add(one, slash(X, X))

        # make slashed and frozen rules
        for p in parent:
            (head, body) = p

            if p not in Ps:
                # frozen base case
                add(p.w, frozen(head), *body)
            else:

                # slash recursive case
                for X in (Xs if filter else (parent.N | parent.V)):
                    add(p.w, slash(head, X), slash(body[0], X), *body[1:])

                # frozen recursive case
                if body[0] not in Xs:
                    add(p.w, frozen(head), frozen(body[0]), *body[1:])

        # recovery rules
        for Y in parent.N - Xs:
            add(one, Y, frozen(Y))

        for Y in parent.N:
            for X in Xs:
                add(one, Y, frozen(X), slash(Y, X))

    def mapping(self, d):

        f = self.mapping
        frozen = self._frozen
        slash = self._slash

        if not isinstance(d, Derivation):
            assert self.is_terminal(d)
            return d

        elif d.r not in self.Ps:
            # frozen base case
            rest = map(f, d.ys)
            if d.x in self.Xs:
                return tree(d.x, tree(frozen(d.x), *rest), tree(slash(d.x, d.x)))
            else:
                return tree(d.x, tree(frozen(d.x), *rest))

        else:
            dd = f(d.ys[0])
            rest = map(f, d.ys[1:])

            # special handling for the case of a terminal
            if not isinstance(dd, Derivation):
                o = dd
                if o in self.Xs:
                    dd = tree(o, o, tree(slash(o, o)))
                else:
                    dd = tree(o, o)

            if len(dd.ys) == 1:   # frozen
                [o] = dd.ys
                # slash base case; this is the bottommost element of Xs along the spine.
                if d.x in self.Xs:
                    return tree(d.x, tree(frozen(d.x), o, *rest), tree(slash(d.x, d.x)))
                else:
                    return tree(d.x, tree(frozen(d.x), o, *rest))

            else:
                [o, s] = dd.ys
                name = (o.x.X if isinstance(o.x, Frozen) else o.x) if isinstance(o, Derivation) else o
                return tree(d.x, o, tree(slash(d.x, name), s, *rest))


class GLCT(CFG):

    def __init__(self, parent, Xs, Ps, filter, id):
        assert set(Ps) <= set(parent.rules)
        assert all(len(r.body) > 0 for r in Ps)

        super().__init__(R=parent.R, S=parent.S, V=set(parent.V))

        self.Xs = Xs
        self.Ps = Ps
        self.filter = filter
        self.id = id
        self.parent = parent

        # TODO: to ensure fresh symbols, use the following.
        #slash = lambda X,Y: Slash(X,Y,id)
        #frozen = lambda X: X if self.is_terminal(X) else Frozen(X,id)

        slash = self._slash; frozen = self._frozen; one = self.R.one
        add = self.add

        Xs = set(Xs)

        if filter:

            # `retained` is the set of symbols that appear outside the
            # left-corner paths. These items may need recovery rules.
            retained = {parent.S}
            for p in parent:
                for X in p.body[int(p in Ps):]:
                    if parent.is_nonterminal(X):
                        retained.add(X)

            # Left corner graph over symbols, but only the rules in Ps.
            G = parent._left_recursion_graph(Ps).reverse()
            T = nx.transitive_closure(G, reflexive=True)

            # `den2num` represents {Y: (den ⇝ num) the left edge}
            den2num = {den: {num for _, num in T.edges(den)} for den in parent.N | parent.V}

            # below is the set of (retained) numerators that are reachable from the denominators
            useful_num = {num for den in Xs for num in den2num[den] if num in retained}

            # In GLCT, we create a rule for each possible consumer of the left corner
            num_given_den = lambda den: (den2num.get(den, set()) & retained)

            # den in Xs ~~> mid ~~~> num in retained
            useful_mid = {mid
                          for den in Xs
                          for mid in den2num[den]
                          for num in den2num[mid]
                          if num in retained}

        else:
            retained = parent.N
            num_given_den = lambda _: parent.N
            useful_num = parent.N | parent.V
            useful_mid = parent.N | parent.V

        # base case
        for X in useful_num:
            add(one, slash(X, X))

        # make slashed and frozen rules
        for p in parent:
            (head, body) = p
            if p not in Ps:
                add(p.w, frozen(head), *body)
            else:
                for Y in num_given_den(body[0]):
                    if body[0] not in useful_mid: continue
                    add(p.w, slash(Y, body[0]), *body[1:], slash(Y, head))
                if body[0] not in Xs:
                    add(p.w, frozen(head), frozen(body[0]), *body[1:])

        # recovery rules
        for Y in retained - Xs:
            add(one, Y, frozen(Y))
        for X in Xs:
            for Y in num_given_den(X):
                add(one, Y, frozen(X), slash(Y, X))

    @cached_property
    def _speculation(self):
        return self.parent.speculate(Xs=self.Xs, Ps=self.Ps, filter=self.filter, id=self.id)

    def mapping(self, d):
        # Our implementation uses the speculation mapping followed by a
        # transpose mapping on the slashed items.
        return self._mapping(self._speculation.mapping(d))

    def _mapping(self, d):
        "Helper method; transposes the slash items."
        if not isinstance(d, Derivation):
            return d
        elif isinstance(d.x, Slash):
            spine = []
            rests = []
            curr = d
            while len(curr.ys) != 0:
                assert isinstance(curr.x, Slash)
                spine.append(curr.ys[0].x)
                rests.append(tuple(map(self._mapping, curr.ys[1:])))
                curr = curr.ys[0]
            num = d.x.Y
            new = tree(self._slash(num, num))
            for rest, s in zip(rests, spine):
                new = tree(self._slash(num, s.Y), *rest, new)
            return new
        else:
            return tree(d.x, *map(self._mapping, d.ys))

    def elim_nullary_slash(self, binarize=True):
        """
        Optimized method for eliminating nullary rules created by the
        left-corner and speculation transformations; should match `nullaryremove`.
        """
        if binarize: self = self.binarize()

        W = self.R.chart()
        v = self.R.chart()

        for p in self:
            head, body = p

            # unary slash
            if len(body) == 1 and isinstance(head, Slash):
                assert isinstance(body[0], Slash)
                W[head, body[0]] += p.w

            # nullary slash
            if len(body) == 0 and isinstance(head, Slash):
                v[head] += p.w

            # This optimized method assumes that the grammar prior to
            # transformation is nullary free.  If the assertion below fails,
            # then so does that assumption.
            assert not len(body) == 0 or isinstance(head, Slash), p

        K = self._lehmann(self.N, W)

        null_weight = self.R.chart()
        for X in self.N:
            for Y in self.N:
                null_weight[X] += K[X,Y] * v[Y]

        return self._push_null_weights(null_weight)


def tree(x, *ys):
    r = Rule(None, x, tuple(label(y) for y in ys))
    return Derivation(r, x, *ys)


def label(d):
    return d.x if isinstance(d, Derivation) else d
