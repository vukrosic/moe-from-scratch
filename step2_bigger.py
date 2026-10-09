"""Step 2. The obvious fix: make the middle layer four times wider."""
import moe

small, big = moe.trained('dense', hid=32), moe.trained('dense', hid=128)
print(f'{"":14}{"numbers":>9}{"work per point":>16}{"correct":>10}')
for name, n in (('small', small), ('four times', big)):
    print(f'{name:14}{n.numbers():9,}{n.work():16,}{moe.score(n)[0]:9.1f} %')
print(f'\nThe wide network does {big.work() / small.work():.0f} times the multiplications for every single point.')
