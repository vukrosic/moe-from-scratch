"""Step 1. One small network paints all sixteen patterns."""
import moe

net = moe.trained('dense', hid=32)
total, per = moe.score(net)
print('The pattern called "target", as it should be:')
print(moe.show(8))
print('\nAs the small network paints it:')
print(moe.show(8, net))
print(f'\nNumbers in the middle layer: {net.numbers():,}')
print(f'Multiplications for one point: {net.work():,}')
print(f'Points painted correctly, all sixteen pages: {total:.1f} %')
for name, p in sorted(zip(moe.NAMES, per), key=lambda x: x[1]):
    print(f'  {name:9} {p:5.1f} %')
