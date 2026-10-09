"""Step 3. Cut the wide layer into eight small experts. A router sends each point to two of them."""
import numpy as np
import moe

net = moe.trained('moe', E=8, K=2)
X, t = moe.page(8, 64)
one = X[2080:2081]                       # one point of the target page
s, c = net.forward(one)
print('Router scores for one point of the target page:')
print('  ' + '  '.join(f'{v:+.2f}' for v in c['g'][0]))
print('Chosen experts:', [int(e) for e in np.where(c['mask'][0])[0]], ' their shares:', [round(float(v), 2) for v in c['pr'][0][c['mask'][0]]])
big, small = moe.trained('dense', hid=128), moe.trained('dense', hid=32)
print(f'\n{"":14}{"numbers":>9}{"work per point":>16}{"correct":>10}')
for name, n in (('small', small), ('four times', big), ('eight experts', net)):
    print(f'{name:14}{n.numbers():9,}{n.work():16,}{moe.score(n)[0]:9.1f} %')
print(f'\nWork of the experts: {net.D} x {net.E} for the router + 2 experts x {2 * net.D * net.hid:,} = {net.work():,}')
