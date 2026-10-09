"""Checks for moe.py. Run: python3 tests.py"""
import numpy as np
import moe


def numeric(net, X, t, key, idx, eps=1e-5, balance=0.0):
    def loss():
        s, c = net.forward(X)
        y = 1 / (1 + np.exp(-s))
        L = -np.mean(t * np.log(y + 1e-9) + (1 - t) * np.log(1 - y + 1e-9))
        if balance:
            q = np.exp(c['g'] - c['g'].max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            L += balance * net.E * (q.mean(0) ** 2).sum()
        return L
    old = net.p[key][idx]
    net.p[key][idx] = old + eps; a = loss()
    net.p[key][idx] = old - eps; b = loss()
    net.p[key][idx] = old
    return (a - b) / (2 * eps)


def analytic(net, X, t, balance=0.0):
    """The gradient that step() computes, recovered from a tiny plain step without Adam history."""
    import copy
    n2 = copy.deepcopy(net)
    before = {k: v.copy() for k, v in n2.p.items()}
    n2.t = 0
    n2.step(X, t, lr=1e-3, balance=balance)
    # after the first Adam step every number moves by lr * sign(gradient)
    return {k: np.sign(before[k] - n2.p[k]) for k in before}


def test_patterns_have_ink_and_paper():
    for j in range(moe.JOBS):
        share = moe.page(j)[1].mean()
        assert 0.1 < share < 0.9, (moe.NAMES[j], share)


def test_every_point_goes_to_k_experts():
    for K in (1, 2, 3):
        net = moe.Net('moe', E=8, K=K)
        X, _, _ = moe.batch(500, np.random.default_rng(0))
        assert (net.routes(X).sum(1) == K).all()
        s, c = net.forward(X)
        assert np.allclose(c['pr'].sum(1), 1)


def test_work_count():
    assert moe.Net('dense', hid=32).work() == 2 * 32 * 32
    assert moe.Net('dense', hid=128).work() == 4 * moe.Net('dense', hid=32).work()
    assert moe.Net('moe', E=8, K=2).work() == 32 * 8 + 2 * 2 * 32 * 16
    assert moe.Net('moe', E=32, K=2).work() - moe.Net('moe', E=8, K=2).work() == 32 * 24      # only the router grows


def test_one_expert_is_a_plain_network():
    m = moe.Net('moe', E=1, K=1, hid=16, seed=3)
    d = moe.Net('dense', hid=16, seed=4)
    for k in ('A', 'a', 'O', 'o'): d.p[k] = m.p[k]
    d.p['W1'], d.p['b1'], d.p['W2'], d.p['b2'] = m.p['W1'][0], m.p['b1'][0], m.p['W2'][0], m.p['b2'][0]
    X, _, _ = moe.batch(200, np.random.default_rng(1))
    assert np.allclose(m.forward(X)[0], d.forward(X)[0])


def test_gradient_signs_match_numbers():
    rng = np.random.default_rng(5)
    for kind, bal in (('dense', 0.0), ('moe', 0.0), ('moe', 0.3)):
        net = moe.Net(kind, E=4, K=2, hid=8, seed=2)
        X, t, _ = moe.batch(64, rng)
        g = analytic(net, X, t, bal)
        for key in net.p:
            flat = rng.integers(0, net.p[key].size, 4)
            for i in flat:
                idx = np.unravel_index(i, net.p[key].shape)
                num = numeric(net, X, t, key, idx, balance=bal)
                if abs(num) > 1e-6:
                    assert np.sign(num) == g[key][idx], (kind, bal, key, idx, num)


def test_training_learns():
    net = moe.Net('moe', E=8, K=2)
    before = moe.score(net, 32)[0]
    moe.train(net, 600)
    assert moe.score(net, 32)[0] > before + 10


def test_balance_rule_evens_the_shares():
    a, b = moe.Net('moe', E=16, K=2), moe.Net('moe', E=16, K=2)
    moe.train(a, 1500); moe.train(b, 1500, balance=0.3)
    assert moe.usage(b, 32)[0].min() > moe.usage(a, 32)[0].min()
    assert moe.usage(b, 32)[0].max() < moe.usage(a, 32)[0].max()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for f in tests:
        f(); print('ok  ', f.__name__)
    print(f'{len(tests)} checks passed')
