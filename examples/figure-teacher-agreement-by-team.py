"""Draw each team's share of its labelled rows where the teacher agrees with gold, from highest to lowest, against the overall share.
The printed lines give the overall agreement, which equals teacher-agreement's 549/1000, and how few or many rows each team has."""

from lib.figures import ACCENT, INK, new_figure, save

import csv  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from matplotlib.ticker import PercentFormatter  # noqa: E402

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
    agree[gold] += int(row["label"] == gold)  # the teacher's team index, compared as teacher-agreement compares it

total, total_agree = len(teacher_rows), sum(agree.values())
assert (total_agree, total) == (549, 1000), "overall agreement differs from teacher-agreement"  # its line
share = sorted(((agree[t] / rows[t], t) for t in rows), reverse=True)
below_half = sum(s < 0.5 for s, _ in share)

print(f"overall agreement, all labelled rows: {total_agree}/{total} ({total_agree / total:.1%})")
print(f"teams with labelled rows: {len(share)}")
print(f"labelled rows per team, fewest: {min(rows.values())}")
print(f"labelled rows per team, most: {max(rows.values())}")
print(f"highest team agreement: {share[0][0]:.1%}")
print(f"lowest team agreement: {share[-1][0]:.1%}")
print(f"teams with agreement below one half: {below_half}")

fig, ax = new_figure()
ax.bar(range(1, len(share) + 1), [s for s, _ in share], color=ACCENT,
       label="each team: share of its rows the teacher gets right")
ax.axhline(total_agree / total, color=INK, linestyle="--", label=f"all teams together: {total_agree / total:.1%}")
ax.set_xlabel("team, from highest to lowest agreement")
ax.set_ylabel("share of rows agreeing with gold")
ax.set_xlim(0, len(share) + 1)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
save(fig, "teacher-agreement-by-team")
