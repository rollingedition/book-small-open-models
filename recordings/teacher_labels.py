#!/usr/bin/env python3
"""Recording teacher-labels: the stand-in labels 1,000 training rows (not dev, not test). Run once.

The rows are drawn from the training rows not in dev, with random.Random(20261112).sample, and sorted.
Same stand-in and settings as standin_baseline.py. Reads train.csv only.
Writes examples/data/recorded/teacher-labels/labels.jsonl (resumable) and manifest.json.
Usage: python recordings/teacher_labels.py --gguf PATH [--limit N] [--out DIR]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

NAME = "teacher-labels"
DRAWN = 2000
ROWS = 1000  # cut from 2,000 on Nov 12: the first 1,000 of the seeded draw, a random half; see the plan


def teacher_rows(limit=None):
    """1,000 training rows, not in dev: the first 1,000 of a seeded draw of 2,000 (a random subset). Sorted."""
    dev = set(c.rl.read_sample("dev"))
    pool = [i for i in range(len(c.rl.read_rows("train"))) if i not in dev]
    chosen = sorted(random.Random(20261112).sample(pool, DRAWN)[:ROWS])
    return chosen[:limit] if limit else chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", help="path to the GGUF (or set STANDIN_GGUF)")
    ap.add_argument("--limit", type=int, help="first N of the 1,000 rows only (smoke runs)")
    ap.add_argument("--out", help="output folder (default: examples/data/recorded/teacher-labels)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else c.RECORDED / NAME
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "labels.jsonl"
    rows = c.rl.read_rows("train")
    chosen = teacher_rows(args.limit)
    if jsonl.exists():  # keep only labels for the chosen rows (the draw was cut from 2,000 to 1,000)
        keep = set(chosen)
        lines = [l for l in jsonl.read_text().splitlines() if json.loads(l)["index"] in keep]
        jsonl.write_text("".join(l + "\n" for l in lines))
    todo = [i for i in chosen if i not in c.done_indices(jsonl)]

    c.rl.setup_runtime()
    llm, grammar = c.standin(c.gguf_path(args.gguf))
    with open(jsonl, "a") as f:
        for i in todo:
            label, text, logprob = c.standin_team(llm, grammar, rows[i][0])
            f.write(json.dumps({"index": i, "label": label, "team": text, "logprob": logprob}) + "\n")
            f.flush()

    if args.limit is None:
        c.write_manifest(
            out, "teacher_labels.py", c.STANDIN_REPO + " / " + c.STANDIN_FILE, c.STANDIN_REVISION,
            settings={"gguf_sha256": c.STANDIN_SHA256, "decoding": "greedy, temperature 0", "seed": 42,
                      "n_ctx": 2048, "n_threads": c.THREADS, "max_tokens": 16,
                      "constraint": "GBNF alternatives over the 77 labels", "chat_template": "embedded in the GGUF",
                      "sample": "train rows not in dev: the first 1,000 of random.Random(20261112).sample(pool, 2000), sorted", "messages": ROWS},
            outputs=["labels.jsonl"], prompt_role="input: the router prompt, in the GGUF chat template",
        )
    print("done", len(c.done_indices(jsonl)), "labels")


if __name__ == "__main__":
    main()
