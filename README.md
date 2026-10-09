# Mixture of experts from scratch

Code for the video **How a 671-Billion-Number AI Does So Little Work**.
DeepSeek-V3 stores 671 billion numbers and uses about 37 billion of them for each piece of a word. The design behind that is called a mixture of experts.
These seven small programs build one in plain Python with numpy. The task is a toy: a network paints sixteen black-and-white patterns, 64 × 64 points each, and is judged by the share of points it paints correctly.

```bash
pip install numpy
./run_all.sh          # runs all seven steps and the tests; expected output is in demo-output.txt
```

The first run trains every network and saves it in `files/`. Later runs reuse them.

| File | Idea | What the run shows |
| --- | --- | --- |
| `moe.py` | The shared parts | The sixteen patterns, a plain network, the experts, the router, the balance rule, training, the work count. |
| `step1_small.py` | A small network | 2,112 numbers in the middle layer, 2,048 multiplications for one point, 91.5 % of the points correct. |
| `step2_bigger.py` | Make it four times wider | 8,352 numbers, 96.2 % correct, but 8,192 multiplications for every point. |
| `step3_experts.py` | Split it into eight experts | A router scores the experts and wakes the best two. 8,832 numbers, 2,304 multiplications, 95.2 % correct. |
| `step4_inside.py` | Who painted what | The share of each pattern that goes to each expert. Busiest expert 52 % of the points, laziest 11 %. |
| `step5_asleep.py` | More experts, a new problem | 16 experts: 94.3 % correct, and 6 experts get fewer than 1 point in 100. 6,624 numbers stored and never used. |
| `step6_balance.py` | A balance rule | A penalty when the router favours a few experts. 16 experts: 95.2 % correct, none asleep. |
| `step7_scale.py` | 2 to 32 experts | 32 experts store 35,328 numbers, do 3,072 multiplications and score 96.1 %: 38 % of the wide network's work for the same score. |
| `tests.py` | Checks of the parts | 7 tests. |

## What this is not

The networks are toys with a few thousand numbers. Real models put a set of experts in every layer and route each piece of a word, not each point of a picture.
A mixture of experts cuts the work, not the memory. Every expert stays loaded, so 32 experts need 4.2 times the memory of the wide network here.
The score stops rising after 8 experts on this task, and at 32 experts 6 are asleep again even with the rule on.

## Try it

`python3 exercise.py` — wake only one expert for each point. The work drops from 2,304 to 1,280 multiplications and the score from 95.5 % to 92.7 %.
Then switch the balance rule on (`RULE = 0.3`). The questions are at the top of the file.

MIT licence.
