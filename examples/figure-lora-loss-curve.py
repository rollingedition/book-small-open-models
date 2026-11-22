"""Draw the loss at each of the recording's steps, with its moving mean, against the uniform guess over 77 teams.
The printed means of steps 1-30 and of the last 100 steps are the two points the prose rests on."""

import json
import math
from pathlib import Path

import numpy as np

from lib.figures import ACCENT, INK, MUTED, new_figure, save

RECORDED = Path(__file__).resolve().parent / "data" / "recorded" / "lora-full"
WINDOW = 25  # steps in the moving mean; each point averages its step and the 24 before it
rows = [json.loads(line) for line in (RECORDED / "steps.jsonl").read_text().splitlines()]
steps = np.array([r["step"] for r in rows])
loss = np.array([r["loss"] for r in rows])
mean = np.convolve(loss, np.ones(WINDOW) / WINDOW, mode="valid")  # mean[k] covers steps k+1 to k+WINDOW
uniform = math.log(77)

print(f"steps in the recording: {len(loss)}")
print(f"window of the moving mean: {WINDOW} steps")
print(f"mean loss of steps 1-30: {loss[:30].mean():.2f}")
print(f"mean loss of the last 100 steps: {loss[-100:].mean():.2f}")
print(f"loss of a uniform guess over 77 teams: {uniform:.2f}")
assert f"{uniform:.2f}" == "4.34"  # the uniform line lora-thirty-steps prints
short_run = {10: 4.13, 20: 6.12, 30: 4.38}  # the losses lora-thirty-steps prints at those steps
same = all(round(loss[s - 1], 2) == v for s, v in short_run.items())
assert same
print(f"recording's losses at steps 10, 20, 30 equal the short run's: {'yes' if same else 'no'}")

fig, ax = new_figure(height=3.2)
ax.plot(steps, loss, color=MUTED, linewidth=0.5, label="loss at each step")
ax.plot(steps[WINDOW - 1:], mean, color=ACCENT, linewidth=1.8, label=f"mean of the last {WINDOW} steps")  # the first mean ends at step WINDOW, so it starts there
ax.axhline(uniform, color=INK, linestyle="--", linewidth=1.1, label="uniform guess over 77 teams, ln 77")
ax.axvspan(1, 30, color=MUTED, alpha=0.15, linewidth=0)
ax.text(34, 8.5, "the short run's 30 steps", ha="left", color=INK)
ax.set_xlabel("training step")
ax.set_ylabel("loss, in nats")
ax.set_ylim(0, 9)
ax.legend(frameon=False, loc="upper right")
save(fig, "lora-loss-curve")
