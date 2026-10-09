"""Exercise. In the film every point goes to TWO experts. What if it goes to only one?

Run this file as it is first (it trains two models, about a minute). Then answer:

  1. How much work per point does one expert save, compared with two?
  2. What does it cost in correct points?
  3. How many experts fall asleep? Print the shares and look.
  4. Now switch the balance rule on for the one-expert model: change RULE below to 0.3 and run again.
     Does the rule help here as it helped with two experts? Write down what you measured before you explain it.
  5. Harder: in moe.py, method Net.forward, the router picks the K highest scores. During training only,
     add a little random noise to the scores before the pick (np.random.normal(0, 0.5, g.shape)).
     Sleeping experts now get a few points by chance. Does that wake them up?
"""
import moe

RULE = 0.0          # question 4: change to 0.3

two = moe.trained('moe', E=8, K=2, balance=0.3)
one = moe.trained('moe', E=8, K=1, balance=RULE)
print(f'{"":22}{"work per point":>16}{"correct":>10}{"asleep":>8}   share of the points per expert')
for name, n in (('two experts per point', two), ('one expert per point', one)):
    share = moe.usage(n)[0]
    print(f'{name:22}{n.work():16,}{moe.score(n)[0]:9.1f} %{len(moe.asleep(n)):8}   ' + ' '.join(f'{v * 100:.0f}' for v in share))
