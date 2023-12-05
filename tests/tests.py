import numpy as np
from collections import defaultdict
from leftcorner import Boolean, Real, Rule, CFG, Entropy, \
    Boolean, MaxPlus, MaxTimes, Log, Derivation
from leftcorner.misc import timeit, colors, powerset, assert_equal_chart, \
    display_table, load_atis


tol = 1e-5


def test_misc():
    with timeit('testing'):
        pass

    Derivation(None, Derivation(None, 'X'))._repr_html_()

    CFG.from_string('', Real)._repr_html_()

    Derivation.to_nltk(None)

    cfg = CFG.from_string('1: X -> Y', Real)
    cfg.left_recursion_graph()._repr_html_()

    try:
        CFG.from_string('x -> y : 1', Real)
    except ValueError:
        pass

    display_table([[cfg, "hello"], [cfg, cfg]], headings=['a', 'b'])
    display_table([[cfg, "hello"], [cfg, cfg]])

    atis = load_atis('data/atis-grammar.txt')
    assert (357, 192, 4592) == (len(atis.V), len(atis.N), len(atis.rules))

    print(colors.thick_line(5), colors.line(5))

    # include an expected-failure test
    try:
        assert_equal_chart({'a': Real(1)}, {'a': Real(2)})
    except AssertionError:
        pass
    else:
        raise AssertionError('test failed')


def test_semirings():

    p = Entropy.from_string('1')
    assert p.H == 0

    # uniform distrbution over 3 elements
    g = CFG.from_string("""

    .25: S → a
    .25: S → b
    .25: S → c
    .25: S → d
    0.0: S → e

    """, Entropy)

    assert np.allclose(g.treesum(tol=tol).H, 2.0)

    z = Entropy.zero
    e = Entropy.one
    x = Entropy.from_string('0.5')
    y = Entropy.from_string('0.2')
    a = Entropy.from_string('0.1')

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star() == e + x * x.star()
    assert x.star() == e + x.star() * x

    assert ((x + y) * a).metric(x * a + y * a) <= 1e-10

    g = CFG.from_string("""
    1: A → a
    0: B → b
    """, Boolean)

    assert_equal_chart(g.agenda(), {
        'a': Boolean(True),
        'b': Boolean(True),
        'A': Boolean(True),
    })

    z = Boolean.zero
    e = Boolean.one
    x = Boolean.from_string('True')
    y = Boolean.from_string('False')

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star() == e + x * x.star()
    assert x.star() == e + x.star() * x

    for a in [z, e]:
        for b in [z, e]:
            for c in [z, e]:
                assert (a + b) * c == a * c + b * c

    z = MaxPlus.zero
    e = MaxPlus.one
    x = MaxPlus.from_string('-3')
    y = MaxPlus.from_string('-4')
    w = MaxPlus.from_string('-5')

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star() == e + x * x.star()
    assert x.star() == e + x.star() * x

    for a in [w,x,y]:
        for b in [w,x,y]:
            for c in [w,x,y]:
                assert (a + b) * c == a * c + b * c


    z = MaxTimes.zero
    e = MaxTimes.one
    x = MaxTimes.from_string('.3')
    y = MaxTimes.from_string('.2')
    w = MaxTimes.from_string('.1')

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star() == e + x * x.star()
    assert x.star() == e + x.star() * x

    for a in [w,x,y]:
        for b in [w,x,y]:
            for c in [w,x,y]:
                assert (a + b) * c == a * c + b * c


    z = Log.zero
    e = Log.one
    x = Log.from_string('-3')
    y = Log.from_string('-2')
    w = Log.from_string('-1')

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star().metric(e + x * x.star()) <= 1e-10
    assert x.star().metric(e + x.star() * x) <= 1e-10

    for a in [w,x,y]:
        for b in [w,x,y]:
            for c in [w,x,y]:
                assert ((a + b) * c).metric(a * c + b * c) <= 1e-10



def test_treesum():

    cfg = CFG.from_string("""

    0.25: S → S S
    0.75: S → a

    """, Real)

    want = cfg.naive_bottom_up()
    have = cfg.agenda(tol=tol)

    for x in want.keys() | have.keys():
        #print(x, want[x].score, have[x].score)
        assert abs(want[x].score - have[x].score) <= 0.0001 * abs(want[x].score)

    # run for fewer iterations
    have = cfg.naive_bottom_up(timeout=2)
    have = {str(k): v.score for k,v in have.items()}
    want = {'a': 1.0, 'S': 0.890625}
    assert have == want, [have, want]


def test_trim():

    cfg = CFG.from_string("""

    0.25: S → S S
    0.75: S → a

    0.75: A → a

    1: C → D
    1: D → C

    1: B → a
    1: B → B

    """, Real)

    have = cfg.trim()

    want = CFG.from_string("""

    0.25: S → S S
    0.75: S → a

    """, Real)

    have.assert_equal(want)

    # cotrim keeps rules that build stuff bottom-up but aren't necessarily used by S.
    have = cfg.cotrim()
    want = CFG.from_string("""

    0.25: S → S S
    0.75: S → a

    1:    B → B
    1:    B → a
    0.75: A → a

    """, Real)
    have.assert_equal(want)


def test_cnf():

    cfg = CFG.from_string("""

    1: S → S1

    1: S → A B C d

    0.5: S1 → S1

    0.1: S1 →
    0.1: A →

    1: A → a
    1: B → d
    1: C → c

    """, Real)

    cnf = cfg.cnf()
    print(cnf)

    assert not cfg.in_cnf()
    assert cnf.in_cnf()

    #print(cnf.treesum().score, cfg.treesum().score)

    assert abs(cnf.treesum().score - cfg.treesum().score) <= 1e-10


def test_grammar_size_metrics():

    cfg = CFG.from_string("""

    1.0: S → A B C D

    0.5: S → S

    0.2: S →

    0.1: A →

    1: A → a
    1: B → d
    1: C → c
    1: D → d

    """, Real)

    assert cfg.size == 17
    assert cfg.num_rules == 8


def test_palindrome_derivations():
    cfg = CFG.from_string("""

    1: S → a S a
    1: S → b S b
    1: S → c

    """, Real)

    s = 'a b c b a'.split()

    n = 0
    print(colors.yellow % 'Derivations:', s)
    for t in cfg.derivations_of(s):
        print(colors.orange % 'derivation:', t)
        assert t.weight().score == 1
        n += 1
    assert n == 1

    n = 0
    print(colors.yellow % 'Derivations:')
    for t in cfg.derivations(cfg.S, 5):
        print(colors.orange % 'derivation:', t)
        n += 1
    assert n == 31, n

    #W = total(cfg, s)
    #print(colors.yellow % 'total weight:', W)
    #assert W.score == 1


def test_catalan_derivations():

    cfg = CFG.from_string("""

    0.25: S → S S
    0.75: S → a

    """, Real)

    # apply left-recursion elimination
    Ps = cfg.find_lr_rules()
    Xs = {p.head for p in Ps}
    cfg = cfg.elim_left_recursion().nullaryremove().trim()
    assert not cfg.is_left_recursive()

    print(cfg)

    s = 'a a a a'.split()

    print(colors.light.yellow % 'Derivations of', s)
    n = 0
    for t in cfg.derivations_of(s):
        print(colors.yellow % 'derivation:', t)
        n += 1
        assert t.Yield() == s
        assert abs(t.weight().score - 0.00494384765625) <= 1e-10

    assert n == 5

    #W = total(cfg, s)
    #print(colors.yellow % 'total weight:', W)
    #assert np.allclose(W.score, 0.02471923828125)


def test_fast_slash_nullary_elim_geom():

    cfg = CFG.from_string("""
      1: X -> b
     .5: X -> X a
    """, start='X', semiring=Real)

    lc = cfg.lc_generalized(Ps=cfg.rules, Xs=cfg.N | cfg.V)
    print(lc)

    #print('\nNull-weight equations:')
    #for a in lc.V:
    #    print(f'null[{a}] += 0')
    #for r in lc.rules:
    #    rhs = ' '.join(f'null[{y}]' for y in r.body)
    #    print(f'null[{r.head}] +=', r.w, rhs)

    new = lc.elim_nullary_slash(binarize=False)
    old = lc.nullaryremove(binarize=False)
    new.assert_equal(old)


def test_unfold():

    cfg = CFG.from_string("""
    1.0: S →
    0.5: S → S a
    0.5: B → b
    """, Real)

    p = cfg.rules[1]

    new = cfg.unfold(p, 0)
    print(new)

    assert cfg.treesum(tol=tol).metric(new.treesum(tol=tol)) <= 1e-12

    new.assert_equal(CFG.from_string("""

    1.0: S →
    0.5: S → a
    0.25: S → S a a
    0.5: B → b

    """, Real))

    # unfolding terminals is not allowed
    try:
        new = cfg.unfold(p, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('expected error')


def test_left_recursion():

    cfg = CFG.from_string("""

    1.0: S →
    0.5: S → S a

    """, Real)

    assert cfg.is_left_recursive()

    # apply left-recursion elimination
    Ps = cfg.find_lr_rules()
    Xs = {p.head for p in Ps}
    lcfg = cfg.lc_generalized(Xs, Ps).trim()

    assert not lcfg.is_left_recursive()


def test_speculation_1():

    cfg = CFG.from_string("""
    1:   S -> B C
    .05: B -> H
    .3:  C -> F C
    .1:  C -> D E X

    1: C -> G
    1: D -> d
    1: E -> e
    1: F -> f
    1: G -> g
    1: H -> h
    1: X -> x
    """, Real)

    Ps = {
        cfg.rules[0],
        cfg.rules[2],
        cfg.rules[3],
    }
    Xs = {'X'}
    scfg = cfg.speculate(Xs, Ps, filter=False)
    assert_equal_chart(cfg.agenda(tol=tol), scfg.agenda(tol=tol), domain=cfg.N, tol=2*tol)

    scfg_t = cfg.speculate(Xs, Ps, filter=False).trim()
    scfg_f = cfg.speculate(Xs, Ps, filter=True)
    assert_equal_chart(scfg_f.agenda(tol=tol), scfg_t.agenda(tol=tol), domain=scfg_t.N, tol=2*tol)


def test_speculation_2():

    cfg = CFG.from_string("""

    1:   S -> X
    1:   X -> B X
    .75: B -> b
    2:   X -> x

    """, Real)

    Ps = {
        cfg.rules[0],
        cfg.rules[1],
    }
    Xs = {'X'}

    scfg = cfg.speculate(Xs, Ps, filter=False)

    assert_equal_chart(cfg.agenda(tol), scfg.agenda(tol), domain=cfg.N, tol=2*tol)

    scfg_t = cfg.speculate(Xs=Xs, Ps=Ps, filter=False).trim()
    scfg_f = cfg.speculate(Xs=Xs, Ps=Ps, filter=True)
    #print()
    #print('spec.trim=')
    #print(scfg_t.trim())

    #print()
    #print('glct=')
    #print(cfg.lc_generalized(Xs=Xs, Ps=Ps).trim())

    assert_equal_chart(scfg_f.agenda(tol=tol), scfg_t.agenda(tol=tol), domain=scfg_t.N, tol=2*tol)


def test_speculation_3():

    cfg = CFG.from_string("""

    .25: A → S
    1:   A → a
    .3:  A → A B
    1:   B → b
    1:   S → A
    1:   S → A B

    """, Real)

    scfg = cfg.speculate(
        Xs={'A', 'S'},
        Ps={
            cfg.rules[4],
            cfg.rules[0],
        },
    ).trim()

    assert_equal_chart(cfg.agenda(tol=tol/10), scfg.agenda(tol=tol/10), domain=cfg.N, tol=1e-3)


def test_glct_1():
    cfg = CFG.from_string("""

    0.5: S → S c
    1: S → A a
    1: S → B b
    1: A → a
    1: B → b

    """, Real)

    assert abs(cfg.treesum(tol=tol).score - 4) <= 1e-8

    Ps = set(CFG.from_string("""
    0.5: S → S c
      1: S → A a
      1: S → B b
    """, Real))

    Xs = {'S', 'A'}
    lcfg = cfg.lc_generalized(Xs, Ps).trim()
    print(lcfg)

    assert abs(lcfg.treesum(tol=tol).score - 4) <= 1e-8


def test_slct_1():
    cfg = CFG.from_string("""

    0.5: S → S c
    1: S → A a
    1: S → B b
    1: A → a
    1: B → b

    """, Real)

    assert abs(cfg.treesum(tol=tol).score - 4) <= 1e-8

    Ps = set(CFG.from_string("""
    0.5: S → S c
      1: S → A a
      1: S → B b
    """, Real))

    s = cfg.lc_selective(Ps).trim()
    assert abs(s.treesum(tol=tol).score - 4) <= 1e-8

    assert not s.is_left_recursive()

    g = cfg.lc_generalized(Ps=Ps, Xs=cfg.N | cfg.V).trim()
    s.assert_equal(g)


def test_slct_2():

    cfg = CFG.from_string("""
     1: S     -> NP VP
    .5: NP    -> PossP NN
    .1: VP    -> VBD
    .2: PossP -> NP POS
    .4: NP    -> PRP NN

    1: PRP -> my
    1: NN  -> sister
    1: POS -> s
    1: NN  -> diploma
    1: VBD -> arrived

    """, Real)

    want = cfg.agenda(tol=tol)

    print('testing without the filter')
    for Ps in powerset(cfg.rules):
        lcfg = cfg.lc_selective(Ps, filter=False)
        have = lcfg.agenda(tol=tol)
        assert_equal_chart(have, want, domain=cfg.N, tol=10*tol)

    print('testing with the filter')
    for Ps in powerset(cfg.rules):
        lcfg = cfg.lc_selective(Ps, filter=True)
        have = lcfg.agenda(tol=tol)
        retained = cfg.N - {'PossP', 'VBD', 'PRP', 'NP'}
        assert_equal_chart(have, want, domain=retained, tol=10*tol)


def test_glct_2():

    cfg = CFG.from_string("""
    1.0: S  → Xk arrived
    0.1: X1 → my
    0.1: X2 → X1 sister
    0.2: X3 → X2 's
    0.5: Xk → X3 diploma
    """, Real)

    t1 = cfg.treesum(tol=tol)

    #print("Original treesum:", t1)

    Xs_ps = list(powerset(list(cfg.N | cfg.V)))
    #Ps_ps = list(powerset(list(cfg._P)))

    # Take 10 random subsets for the test case
    np.random.shuffle(Xs_ps)
    Xs_ps = Xs_ps[:20]

    _Ps = list(powerset(cfg.rules))
    np.random.shuffle(_Ps)
    Ps_ps = _Ps[:20]

    count = 0
    failcount = 0

    for Xs in Xs_ps:
        for Ps in Ps_ps:
            count += 1
            Ps = set(Ps)
            lcfg = cfg.lc_generalized(Xs, Ps) #filter = True)
            t2 = lcfg.treesum(tol=tol)
            if t1.metric(t2) > 2*tol:
                print("FAILS", "\tXs", Xs, "\tPs:", Ps)
                print("treesum:", t1, t2)
                failcount += 1

    for Xs in Xs_ps:
        for Ps in Ps_ps:
            count += 1
            Ps = set(Ps)
            lcfg = cfg.lc_generalized(Xs, Ps, filter=True)
            t2 = lcfg.treesum(tol=tol)
            if t1.metric(t2) > 2*tol:
                print("FAILS", "\tXs", Xs, "\tPs:", Ps)
                print("treesum:", t1, t2)
                failcount += 1

    #print("Fail count:", failcount)
    #print("Fail ratio:", failcount / count)
    #print("Total num cases:", count)
    assert failcount == 0


def test_battery_1():

    cfg = CFG.from_string("""

    1.0: S →
    0.5: S → S a

    """, Real)

    Xs = {cfg.S}
    Ps = [cfg.rules[1]]

    s = cfg.speculate(Xs=Xs, Ps=Ps, filter=False)
    g = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=False)

    C = cfg.agenda(tol=tol)
    S = s.agenda(tol=tol)
    G = s.agenda(tol=tol)

    assert_equal_chart(S, G, tol=2*tol)
    assert_equal_chart(C, G, domain=cfg.N, tol=2*tol)
    assert_equal_chart(C, S, domain=cfg.N, tol=2*tol)


def test_battery_2():

    cfg = CFG.from_string("""
    1:  S     -> NP VP
    .5: NP    -> Poss NN
    .1: VP    -> arrived
    .2: Poss  -> NP s
    .4: NP    -> my NN

    1: NN  -> sister
    1: NN  -> diploma

    """, Real)

    C = cfg.agenda(tol=tol)

    # this test was too slow; so I am using random sampling
    _Ps = list(powerset(cfg.rules))
    np.random.shuffle(_Ps)
    _Ps = _Ps[:20]

    _Xs = list(powerset(cfg.N))
    np.random.shuffle(_Xs)
    _Xs = _Xs[:20]

    for Ps in _Ps:
        for Xs in _Xs:
            Xs = set(Xs)

            s = cfg.speculate(Xs=Xs, Ps=Ps, filter=False)
            g = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=False)

            S = s.agenda(tol=tol)
            G = g.agenda(tol=tol)

            assert_equal_chart(C, G, domain=cfg.N, tol=2.5*tol)
            assert_equal_chart(C, S, domain=cfg.N, tol=2.5*tol)
            assert_equal_chart(S, G, tol=2.5*tol)


def test_battery_3():
    cfg = CFG.from_string("""

    0.5: S → S c
    .75: S → A a
    .1:  S → B b
    1:   A → a
    1:   B → b

    """, Real)

    C = cfg.agenda(tol=tol)

    for Ps in powerset(cfg.rules):
        for Xs in powerset(cfg.N):
            Xs = set(Xs)

            s = cfg.speculate(Xs=Xs, Ps=Ps, filter=False)
            g = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=False)

            S = s.agenda(tol=tol)
            G = s.agenda(tol=tol)

            assert_equal_chart(S, G, tol=2*tol)
            assert_equal_chart(C, G, domain=cfg.N, tol=2*tol)
            assert_equal_chart(C, S, domain=cfg.N, tol=2*tol)



def test_sccs():
    cfg = CFG.from_string("""

    1: S → C1 w

    0.5: C1 → C2 c
    1.0: C2 → C3 c
    1.0: C3 → C4 c
    1.0: C4 → C5 c
    1.0: C5 → C6 c
    1.0: C6 → C1 c

    1.0: C1 → B1 x

    0.5: B1 → B2 b
    1.0: B2 → B3 b
    1.0: B3 → B4 b
    1.0: B4 → B5 b
    1.0: B5 → B6 b
    1.0: B6 → B1 b

    1.0: B1 → A1 y : 1

    0.5: A1 → A2 a
    1.0: A2 → A3 a
    1.0: A3 → A4 a
    1.0: A4 → A1 a

    1.0: A1 → z

    """, Real)

    # apply left-recursion elimination
    Ps = cfg.find_lr_rules()
    Xs = {p.head for p in Ps}
    old = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=False)

    assert not old.trim().is_left_recursive()

    # apply left-recursion elimination
    Ps = cfg.find_lr_rules()

    Xs_new = cfg.sufficient_Xs(Ps)

    print(f'|Xs|: {len(Xs)} -> {len(Xs_new)}')
    assert Xs >= Xs_new

    new = cfg.lc_generalized(Xs=Xs_new, Ps=Ps, filter=False)
    assert not new.trim().is_left_recursive()


def test_derivation_mapping_speculation():

    for cfg in [

            CFG.from_string("""

            1: S → a b c

            """, Real),

            CFG.from_string("""

            1: S → A c
            1: A → a
            1: A → b

            """, Real),

            CFG.from_string("""

            1: S → S1 a
            1: S1 → D
            1: S1 → E
            1: D → d
            1: E → e

            """, Real),

            CFG.from_string("""

            1: S → S1 a1
            1: S1 → S2 a2
            1: S2 → S3 S3
            1: S3 → S4
            1: S4 → a a4
            1: S4 → C a4
            1: C → c

            """, Real),

            CFG.from_string("""

            1: S → S1 S1
            1: S1 → S2 S2
            1: S2 → S3 d
            1: S3 → a
            1: S3 → b

            """, Real)

    ]:

        T = 12
        src = list(cfg.derivations(cfg.S, 12))

        if 1:
            print(colors.line(80))
            Ps = cfg.rules
            Xs = {'D'}

            sp = cfg.speculate(Xs=Xs, Ps=Ps, filter=False)
            tgt = list(sp.derivations(sp.S, T))
            _test_derivation_mapping(sp, src, tgt)


        if 1:
            print(colors.line(80))

            Ps = cfg.rules
            Xs = cfg.N

            sp = cfg.speculate(Xs=Xs, Ps=Ps, filter=False)
            tgt = list(sp.derivations(sp.S, T))
            _test_derivation_mapping(sp, src, tgt)


        if 1:
            print(colors.line(80))

            Ps = cfg.rules
            Xs = {'d'}

            sp = cfg.speculate(Xs=Xs, Ps=Ps, filter=False)
            tgt = list(sp.derivations(sp.S, T))
            _test_derivation_mapping(sp, src, tgt)


        if 1:
            print(colors.line(80))

            Ps = set()
            Xs = cfg.N

            sp = cfg.speculate(Xs=Xs, Ps=Ps, filter=False)
            tgt = list(sp.derivations(sp.S, T))
            _test_derivation_mapping(sp, src, tgt)


        if 0:

            fails = 0
            total = 0

            _Ps = list(powerset(cfg.rules))
            _Xs = list(powerset(cfg.N | cfg.V))

            for _ in range(10):
                print(colors.line(80))
                Ps = set(_Ps[np.random.choice(range(len(_Ps)))])
                Xs = set(_Xs[np.random.choice(range(len(_Xs)))])
                print(Ps, Xs)

                total += 1

                sp = cfg.speculate(Xs=Xs, Ps=Ps, filter=False)
                tgt = list(sp.derivations(sp.S, T))

                try:
                    _test_derivation_mapping(sp, src, tgt)
                except AssertionError as e:
                    print(colors.light.red % e)
                    fails += 1
                else:
                    print(colors.light.green % 'PASS')

                print(f'success rate: {((total-fails)/total*100):.1f}% ({total-fails} / {total})')

            assert fails == 0


def test_derivation_mapping_glct():

    for cfg in [

            CFG.from_string("""

            1: S → a b c

            """, Real),

            CFG.from_string("""

            1: S → A c
            1: A → a
            1: A → b

            """, Real),

            CFG.from_string("""

            1: S → S1 a
            1: S1 → D
            1: S1 → E
            1: D → d
            1: E → e

            """, Real),

            CFG.from_string("""

            1: S → S1 a1
            1: S1 → S2 a2
            1: S2 → S3 S3
            1: S3 → S4
            1: S4 → a a4
            1: S4 → C a4
            1: C → c

            """, Real),

            CFG.from_string("""

            1: S → S1 S1
            1: S1 → S2 S2
            1: S2 → S3 d
            1: S3 → a
            1: S3 → b

            """, Real)

    ]:

        T = 15
        src = list(cfg.derivations(cfg.S, T))

        if 1:
            print(colors.line(80))

            Ps = cfg.rules
            Xs = cfg.N

            sp = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=True)
            tgt = list(sp.derivations(sp.S, T))

            _test_derivation_mapping(sp, src, tgt)


        if 1:
            print(colors.line(80))
            Ps = cfg.rules
            Xs = {np.random.choice(list(cfg.N))}

            sp = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=True)
            tgt = list(sp.derivations(sp.S, T))
            _test_derivation_mapping(sp, src, tgt)


        if 1:
            print(colors.line(80))

            Ps = cfg.rules
            Xs = {np.random.choice(list(cfg.V))}

            sp = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=True)
            tgt = list(sp.derivations(sp.S, T))
            _test_derivation_mapping(sp, src, tgt)


        if 1:
            print(colors.line(80))

            Ps = set()
            Xs = cfg.N

            sp = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=True)
            tgt = list(sp.derivations(sp.S, T))
            _test_derivation_mapping(sp, src, tgt)


        if 1:

            fails = 0
            total = 0

            _Ps = list(powerset(cfg.rules))
            _Xs = list(powerset(cfg.N | cfg.V))

            for _ in range(10):
                print(colors.line(80))
                Ps = set(_Ps[np.random.choice(range(len(_Ps)))])
                Xs = set(_Xs[np.random.choice(range(len(_Xs)))])
                print(Ps, Xs)

                total += 1

                sp = cfg.lc_generalized(Xs=Xs, Ps=Ps, filter=True)
                tgt = list(sp.derivations(sp.S, T))

                try:
                    _test_derivation_mapping(sp, src, tgt)
                except AssertionError as e:
                    print(colors.light.red % e)
                    fails += 1
                else:
                    print(colors.light.green % 'PASS')

                print(f'success rate: {((total-fails)/total*100):.1f}% ({total-fails} / {total})')

            assert fails == 0


def _test_derivation_mapping(q, src, tgt):
    f = q.mapping

    assert len(src) == len(tgt)
    print(colors.yellow % 'Source Derivations:')
    for t in src:
        print(' ', t)

    print(colors.yellow % 'Target Derivations:')
    for t in tgt:
        print(' ', t)

    print(colors.yellow % 'Forward Mapping:')
    ok = True
    mapsto = defaultdict(list)
    for t in src:
        print(colors.orange % ':', t)
        ft = f(t)
        print(colors.mark(ft in tgt), ft)
        ok &= (ft in tgt)
        mapsto[ft].append(t)

        assert t.Yield() == ft.Yield()
        assert t.x == ft.x

    assert all(len(mapsto[ft]) == 1 for ft in tgt)
    assert ok


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
