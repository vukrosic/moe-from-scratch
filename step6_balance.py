"""Step 6. The balance rule: training also punishes uneven shares between the experts."""
import moe

print(f'{"":12}{"rule":>6}{"correct":>10}{"asleep":>8}   share of the points per expert')
for E in (8, 16):
    for bal in (0.0, 0.3):
        n = moe.trained('moe', E=E, K=2, balance=bal)
        share = moe.usage(n)[0]
        print(f'{E:3} experts {"on" if bal else "off":>6}{moe.score(n)[0]:9.1f} %{len(moe.asleep(n)):8}   ' + ' '.join(f'{v * 100:.0f}' for v in share))
