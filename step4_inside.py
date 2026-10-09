"""Step 4. Look inside: which expert paints which pattern? Nobody told the router."""
import moe

net = moe.trained('moe', E=8, K=2)
share, tab = moe.usage(net)
print('Share of each page that goes to each expert (percent). Two experts work on every point, so a row adds up to 200.\n')
print(f'{"":10}' + ''.join(f'{"E" + str(e + 1):>6}' for e in range(net.E)))
for j, name in enumerate(moe.NAMES):
    print(f'{name:10}' + ''.join(f'{v * 100:6.0f}' if v >= 0.005 else '     .' for v in tab[j]))
print(f'{"all pages":10}' + ''.join(f'{v * 100:6.0f}' for v in share))
print(f'\nBusiest expert: {share.max() * 100:.0f} % of all points. Laziest: {share.min() * 100:.0f} %.')
