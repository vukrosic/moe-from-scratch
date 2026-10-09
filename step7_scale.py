"""Step 7. More experts, two per point: the stored numbers grow, the work per point barely moves."""
import moe

big = moe.trained('dense', hid=128)
print(f'{"experts":>8}{"numbers":>9}{"work per point":>16}{"correct":>10}{"asleep":>8}')
for E in (2, 4, 8, 16, 32):
    n = moe.trained('moe', E=E, K=2, balance=0.3)
    print(f'{E:8}{n.numbers():9,}{n.work():16,}{moe.score(n)[0]:9.1f} %{len(moe.asleep(n)):8}')
print(f'{"wide":>8}{big.numbers():9,}{big.work():16,}{moe.score(big)[0]:9.1f} %')
n = moe.trained('moe', E=32, K=2, balance=0.3)
print(f'\n32 experts store {n.numbers() / big.numbers():.1f} times the numbers of the wide network and do {n.work() / big.work() * 100:.0f} % of its work.')
