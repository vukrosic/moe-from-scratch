#!/bin/sh
# Runs the seven steps in order, then the tests. Python 3 with numpy.
# The first run trains every network and saves it in files/; later runs reuse them.
PY="${PYTHON:-python3}"
cd "$(dirname "$0")" || exit 1
for f in step1_small step2_bigger step3_experts step4_inside step5_asleep step6_balance step7_scale tests; do
  echo "--- $f"
  "$PY" "$f.py" || exit 1
done
