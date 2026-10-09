"""Mixture of experts: many small networks and a router that sends each input to a few of them.

The job of every model here: paint sixteen different black-and-white patterns.
An input is a point (x, y) on the page plus which pattern is asked for. The answer is ink or paper.
"""
import numpy as np

np.seterr(divide='ignore', over='ignore', invalid='ignore')   # numpy on Apple chips prints false matmul warnings

NAMES = ['stripes', 'checkers', 'rings', 'disc', 'spiral', 'cross', 'dots', 'diagonal',
         'target', 'windmill', 'waves', 'frame', 'triangle', 'zigzag', 'moon', 'grid']
JOBS = len(NAMES)
IN = 2 + JOBS


def ink(job, x, y):
    """The true pattern: True where the page has ink. x and y run from -1 to 1."""
    r = np.sqrt(x * x + y * y)
    if job == 0: return np.sin(3 * np.pi * x) > 0
    if job == 1: return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) > 0
    if job == 2: return np.sin(4 * np.pi * r) > 0
    if job == 3: return r < 0.6
    if job == 4: return np.sin(2 * np.arctan2(y, x) + 7 * r) > 0
    if job == 5: return (np.abs(x) < 0.25) | (np.abs(y) < 0.25)
    if job == 6: return np.cos(3 * np.pi * x) + np.cos(3 * np.pi * y) > 0.7
    if job == 7: return np.sin(2.5 * np.pi * (x + y)) > 0
    if job == 8: return np.sin(5 * np.pi * np.sqrt((x - 0.4) ** 2 + (y + 0.3) ** 2)) > 0
    if job == 9: return np.sin(4 * np.arctan2(y, x)) > 0
    if job == 10: return np.sin(4 * np.pi * y + 1.5 * np.sin(3 * np.pi * x)) > 0
    if job == 11: return (np.maximum(np.abs(x), np.abs(y)) > 0.45) & (np.maximum(np.abs(x), np.abs(y)) < 0.8)
    if job == 12: return (y < 0.7) & (y > 1.6 * np.abs(x) - 0.7)
    if job == 13: return np.sin(3 * np.pi * (y + 0.7 * np.abs((x * 2) % 1 - 0.5))) > 0
    if job == 14: return (r < 0.75) & (np.sqrt((x - 0.3) ** 2 + y * y) > 0.55)
    return (np.abs(np.sin(2.5 * np.pi * x)) < 0.35) | (np.abs(np.sin(2.5 * np.pi * y)) < 0.35)


def batch(n, rng):
    """n random questions: a point, a pattern number, and the true answer."""
    xy = rng.uniform(-1, 1, (n, 2))
    job = rng.integers(0, JOBS, n)
    X = np.zeros((n, IN)); X[:, :2] = xy; X[np.arange(n), 2 + job] = 1
    t = np.zeros(n)
    for j in range(JOBS):
        m = job == j
        t[m] = ink(j, xy[m, 0], xy[m, 1])
    return X, t, job


def page(job, n=64):
    """Every point of an n by n page for one pattern, as model input, plus the true picture."""
    c = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(c, c)
    X = np.zeros((n * n, IN)); X[:, 0] = x.ravel(); X[:, 1] = y.ravel(); X[:, 2 + job] = 1
    return X, ink(job, x.ravel(), y.ravel()).astype(float)


class Net:
    """Entry layer -> one middle layer -> answer. The middle layer is either one wide network ('dense')
    or E small experts with a router that picks K of them for each input ('moe')."""

    def __init__(self, kind, D=32, hid=16, E=8, K=2, seed=0):
        r = np.random.default_rng(seed)
        self.kind, self.D, self.hid, self.E, self.K = kind, D, hid, E, K
        p = {'A': r.normal(0, 1.0, (IN, D)) * np.sqrt(2 / IN), 'a': np.zeros(D),
             'O': r.normal(0, 1.0, (D, 1)) * np.sqrt(1 / D), 'o': np.zeros(1)}
        if kind == 'dense':
            p['W1'] = r.normal(0, 1.0, (D, hid)) * np.sqrt(2 / D); p['b1'] = np.zeros(hid)
            p['W2'] = r.normal(0, 1.0, (hid, D)) * np.sqrt(1 / hid); p['b2'] = np.zeros(D)
        else:
            p['G'] = r.normal(0, 1.0, (D, E)) * np.sqrt(1 / D)                 # the router
            p['W1'] = r.normal(0, 1.0, (E, D, hid)) * np.sqrt(2 / D); p['b1'] = np.zeros((E, hid))
            p['W2'] = r.normal(0, 1.0, (E, hid, D)) * np.sqrt(1 / hid); p['b2'] = np.zeros((E, D))
        self.p = p
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.t = 0

    # ---- counting ----
    def numbers(self):
        """How many numbers the middle layer stores."""
        return sum(self.p[k].size for k in self.p if k not in 'AaOo')

    def work(self):
        """Multiplications the middle layer does for ONE input."""
        if self.kind == 'dense':
            return 2 * self.D * self.hid
        return self.D * self.E + self.K * 2 * self.D * self.hid

    # ---- forward ----
    def forward(self, X):
        p = self.p
        z = np.maximum(X @ p['A'] + p['a'], 0)
        c = {'X': X, 'z': z}
        if self.kind == 'dense':
            h = np.maximum(z @ p['W1'] + p['b1'], 0)
            F = h @ p['W2'] + p['b2']
            c['h'] = h
        else:
            g = z @ p['G']                                            # one score per expert
            top = np.argpartition(-g, self.K - 1, axis=1)[:, :self.K]
            mask = np.zeros_like(g, bool); mask[np.arange(len(g))[:, None], top] = True
            gm = np.where(mask, g, -np.inf)
            pr = np.exp(gm - gm.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)   # shares of the chosen experts
            F = np.zeros_like(z); c['h'], c['f'] = {}, {}
            for e in range(self.E):
                rows = mask[:, e]
                if not rows.any(): continue
                h = np.maximum(z[rows] @ p['W1'][e] + p['b1'][e], 0)   # only the chosen rows are computed
                f = h @ p['W2'][e] + p['b2'][e]
                F[rows] += pr[rows, e, None] * f
                c['h'][e], c['f'][e] = h, f
            c.update(g=g, mask=mask, pr=pr)
        u = z + F
        s = (u @ p['O'] + p['o'])[:, 0]
        c.update(u=u, s=s)
        return s, c

    def predict(self, X):
        return self.forward(X)[0] > 0

    def routes(self, X):
        """Which experts each input is sent to: a True/False table, inputs by experts."""
        return self.forward(X)[1]['mask']

    # ---- one training step ----
    def step(self, X, t, lr=0.003, balance=0.0):
        p = self.p; B = len(X)
        s, c = self.forward(X)
        y = 1 / (1 + np.exp(-s))
        loss = -np.mean(t * np.log(y + 1e-9) + (1 - t) * np.log(1 - y + 1e-9))
        ds = ((y - t) / B)[:, None]
        g = {'O': c['u'].T @ ds, 'o': ds.sum(0)}
        du = ds @ p['O'].T
        dz = du.copy(); z = c['z']
        if self.kind == 'dense':
            dh = (du @ p['W2'].T) * (c['h'] > 0)
            g['W2'] = c['h'].T @ du; g['b2'] = du.sum(0)
            g['W1'] = z.T @ dh; g['b1'] = dh.sum(0)
            dz += dh @ p['W1'].T
        else:
            mask, pr = c['mask'], c['pr']
            dp = np.zeros_like(pr)
            for k in ('W1', 'b1', 'W2', 'b2'): g[k] = np.zeros_like(p[k])
            for e in c['h']:
                rows = mask[:, e]; h, f = c['h'][e], c['f'][e]
                dp[rows, e] = (du[rows] * f).sum(1)
                df = du[rows] * pr[rows, e, None]
                dh = (df @ p['W2'][e].T) * (h > 0)
                g['W2'][e] = h.T @ df; g['b2'][e] = df.sum(0)
                g['W1'][e] = z[rows].T @ dh; g['b1'][e] = dh.sum(0)
                dz[rows] += dh @ p['W1'][e].T
            dg = pr * (dp - (pr * dp).sum(1, keepdims=True))
            if balance:
                # the balance rule: every expert should get an equal share of the router's attention
                q = np.exp(c['g'] - c['g'].max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
                m = q.mean(0)
                loss += balance * self.E * (m * m).sum()
                dq = np.tile(balance * self.E * 2 * m / B, (B, 1))
                dg += q * (dq - (q * dq).sum(1, keepdims=True))
            g['G'] = z.T @ dg
            dz += dg @ p['G'].T
        dz *= z > 0
        g['A'] = c['X'].T @ dz; g['a'] = dz.sum(0)
        self.t += 1
        for k in p:                                                    # Adam
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            p[k] -= lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)
        return loss


STEPS = 24000      # training rounds for every model in this repo


def train(net, steps=STEPS, seed=1, bs=256, lr=0.003, balance=0.0, log=None, at=(), hook=None):
    """Train on random questions. The learning rate falls to a tenth over the run.
    `hook(net, round)` is called at round 0 and after every round listed in `at`."""
    rng = np.random.default_rng(seed)
    if hook and 0 in at: hook(net, 0)
    for i in range(steps):
        X, t, _ = batch(bs, rng)
        loss = net.step(X, t, lr * (0.1 ** (i / steps)), balance)
        if log and (i + 1) % log == 0: print(f'  round {i + 1:6,}  loss {loss:.3f}')
        if hook and (i + 1) in at: hook(net, i + 1)
    return net


def score(net, n=64):
    """Paint all sixteen pages and count the correct points. Returns (percent correct, list per pattern)."""
    per = []
    for j in range(JOBS):
        X, t = page(j, n)
        per.append(float((net.predict(X) == (t > 0.5)).mean() * 100))
    return sum(per) / JOBS, per


def usage(net, n=64):
    """Share of all points that each expert receives (the shares sum to K), and a patterns-by-experts table."""
    tab = np.zeros((JOBS, net.E))
    for j in range(JOBS):
        tab[j] = net.routes(page(j, n)[0]).mean(0)
    return tab.mean(0), tab


def asleep(net):
    """Experts that receive fewer than 1 point in 100."""
    return [e for e, u in enumerate(usage(net)[0]) if u < 0.01]


def lead(net, X):
    """For each input, the expert with the highest router score."""
    return net.forward(X)[1]['g'].argmax(1)


def trained(kind, hid=16, E=8, K=2, balance=0.0, steps=STEPS):
    """A trained model. The first call trains it (5 to 40 seconds) and keeps it in files/, later calls load it."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    name = f'{kind}-h{hid}' if kind == 'dense' else f'moe-e{E}-k{K}-b{balance}'
    path = os.path.join(here, 'files', f'{name}-{steps}.npz')
    net = Net(kind, hid=hid, E=E, K=K)
    if os.path.exists(path):
        d = np.load(path)
        for k in net.p: net.p[k] = d[k]
        return net
    print(f'  training {name} for {steps:,} rounds ...', flush=True)
    train(net, steps, balance=balance)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **net.p)
    return net


def show(job, net=None, n=32):
    """Print a page in the terminal: the true pattern, or what `net` paints."""
    X, t = page(job, n)
    pic = (net.predict(X) if net else t > 0.5).reshape(n, n)
    return '\n'.join(''.join('##' if v else '  ' for v in row) for row in pic[::2])
