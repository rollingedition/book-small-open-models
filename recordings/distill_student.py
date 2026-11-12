#!/usr/bin/env python3
"""Recording distill-student: the lora-full recipe trained on the teacher's labels (1,000 rows). Run once.

Reads teacher-labels/labels.jsonl and train.csv only. The rows are the teacher's 1,000; no test message is read.
Writes examples/data/recorded/distill-student/adapter/ and steps.jsonl, then manifest.json.
Usage: python recordings/distill_student.py [--limit STEPS] [--out DIR] [--teacher DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

NAME = "distill-student"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="stop after N steps (smoke runs)")
    ap.add_argument("--out", help="output folder (default: examples/data/recorded/distill-student)")
    ap.add_argument("--teacher", help="teacher-labels folder (default: examples/data/recorded/teacher-labels)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else c.RECORDED / NAME
    teacher = Path(args.teacher) if args.teacher else c.RECORDED / "teacher-labels"
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    rows = c.rl.read_rows("train")
    pairs = [json.loads(line) for line in open(teacher / "labels.jsonl")]
    assert args.limit or len(pairs) == 1000, len(pairs)
    picked = [p["index"] for p in pairs]
    labels = [p["label"] for p in pairs]
    assert len(set(picked)) == len(picked)

    c.rl.setup_runtime()
    import torch
    torch.set_num_threads(c.THREADS)
    losses = c.train_lora([rows[i][0] for i in picked], labels, out / "adapter", args.limit)
    with open(out / "steps.jsonl", "w") as f:
        for step, loss in enumerate(losses, 1):
            f.write(json.dumps({"step": step, "loss": loss}) + "\n")

    if args.limit is None:
        c.write_manifest(
            out, "distill_student.py", c.BASE_REPO, c.pinned_revision(c.BASE_REPO),
            settings={"recipe": "same as lora-full", "labels_from": "teacher-labels", "rows": len(picked),
                      "steps": len(losses), "batch_size": 8, "epochs": 1, "seed": 0, "n_threads": c.THREADS,
                      "dtype": "float32", "max_length": 128},
            outputs=["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "steps.jsonl"],
            prompt_role="not used: the classifier reads the raw message text",
            extra={"adapter": "banking77-distilled-v1"},
        )
    print("done", len(losses), "steps")


if __name__ == "__main__":
    main()
