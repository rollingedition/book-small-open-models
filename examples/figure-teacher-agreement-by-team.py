"""Draw each team's labelled training rows as one bar, the rows the teacher gets right at the bottom and the rest hatched above, teams sorted from the highest share right to the lowest.
The printed lines give the teams where the teacher gets no row right, and the teams below one half, which the prose's point rests on."""

from lib.figures import ACCENT, MUTED, new_figure, save

import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402

torch.set_num_threads(2)

import router_lib  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
labels = json.loads((DATA / "banking77" / "categories.json").read_text())
team_index = {name: i for i, name in enumerate(labels)}
with open(DATA / "banking77" / "train.csv", newline="") as f:
    train = [team_index[row["category"]] for row in csv.DictReader(f)]  # gold team index of each training row
teacher_rows = [json.loads(line) for line in (DATA / "recorded" / "teacher-labels" / "labels.jsonl").read_text().splitlines()]

rows, agree = Counter(), Counter()
for row in teacher_rows:
    gold = train[row["index"]]  # row["index"] is the row's position in the training file
    rows[gold] += 1
    agree[gold] += int(row["label"] == gold)

# teacher-agreement's count, read the way it reads the training file: its line "teacher agrees with gold"
trained = router_lib.read_rows("train")
recomputed = sum(row["label"] == trained[row["index"]][1] for row in teacher_rows)
total, total_agree = len(teacher_rows), sum(agree.values())
assert total_agree == recomputed, "agreement differs from teacher-agreement's count"

order = [t for _, t in sorted(((agree[t] / rows[t], t) for t in rows), reverse=True)]
zero = [t for t in order if agree[t] == 0]
below = [t for t in order if agree[t] / rows[t] < 0.5]
for t in below:  # the teacher is wrong on more than half of each such team's rows
    assert (rows[t] - agree[t]) * 2 > rows[t], "a team below one half is not wrong on most rows"
below_rows = sum(rows[t] for t in below)
below_wrong = sum(rows[t] - agree[t] for t in below)

print(f"overall agreement, all labelled rows: {total_agree}/{total} ({total_agree / total:.1%})")
print(f"teams with labelled rows: {len(order)}")
print(f"labelled rows per team, fewest: {min(rows.values())}")
print(f"labelled rows per team, most: {max(rows.values())}")
print(f"teams where the teacher gets no row right: {len(zero)}")
print(f"rows in the smallest of those teams: {min(rows[t] for t in zero)}")
print(f"teams with agreement below one half: {len(below)}")
print(f"rows in the smallest team below one half: {min(rows[t] for t in below)}")
print(f"rows in teams with agreement below one half: {below_rows}")
print(f"rows in those teams the teacher gets wrong: {below_wrong}")

right = [agree[t] for t in order]
wrong = [rows[t] - agree[t] for t in order]
fig, ax = new_figure()
ax.bar(range(1, len(order) + 1), right, color=ACCENT, label="teacher agrees with gold")
ax.bar(range(1, len(order) + 1), wrong, bottom=right, facecolor="none", edgecolor=MUTED, hatch="////", linewidth=0.6,
       label="teacher disagrees with gold")
ax.set_xlabel("team, from highest to lowest share the teacher gets right")
ax.set_ylabel("training rows labelled by the teacher")
ax.set_xlim(0, len(order) + 1)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
save(fig, "teacher-agreement-by-team")
