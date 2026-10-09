"""Step 5. Sixteen experts instead of eight. More numbers, same work. The score goes down."""
import moe

e8, e16 = moe.trained('moe', E=8, K=2), moe.trained('moe', E=16, K=2)
for name, n in (('8 experts', e8), ('16 experts', e16)):
    share = moe.usage(n)[0]
    print(f'{name:11} numbers {n.numbers():7,}  work {n.work():5,}  correct {moe.score(n)[0]:.1f} %  asleep {len(moe.asleep(n))}')
    print('   share of the points per expert: ' + ' '.join(f'{v * 100:.0f}' for v in share))
dead = moe.asleep(e16)
print(f'\n{len(dead)} of 16 experts get fewer than 1 point in 100: numbers {len(dead) * (e16.numbers() // 16):,} that are stored and never used.')
