"""Draw the messages a second that the throughput recording measured for four setups, on a log scale.
The printed rates are the points the prose rests on; the recording names the machine they were measured on."""

import json
from pathlib import Path

from lib.figures import ACCENT, INK, MUTED, new_figure, save

RECORDED = Path(__file__).resolve().parent / "data" / "recorded" / "throughput"
rates = json.loads((RECORDED / "rates.json").read_text())

setups = [  # top to bottom in the figure
    ("stand-in, one message per call", rates["standin_one_message_per_call"], MUTED),
    ("float32 classifier, one message per call", rates["classifier_fp32_one_message_per_call"], INK),
    ("float32 classifier, batches of 16", rates["classifier_fp32_batches_of_16"], INK),
    ("int8 classifier, batches of 16", rates["classifier_int8_batches_of_16"], ACCENT),
]

print(f"machine: {rates['machine']}")
for name, rate, _ in setups:
    print(f"{name}, messages a second: {rate}")
print(f"int8 over fp32, batches of 16, from these rates: {rates['classifier_int8_batches_of_16'] / rates['classifier_fp32_batches_of_16']:.1f}")

fig, ax = new_figure(height=2.8)
for row, (name, rate, color) in enumerate(setups):
    ax.barh(row, rate, height=0.6, color=color)
    ax.text(rate * 1.12, row, f"{rate}", va="center", ha="left", color=INK)
ax.set_xscale("log")
ax.set_xlim(0.1, 300)
ax.set_yticks(range(len(setups)), [name for name, _, _ in setups])
ax.invert_yaxis()
ax.set_xlabel("messages a second (log scale)")
ax.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
save(fig, "recorded-throughput")
