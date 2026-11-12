#!/usr/bin/env python3
"""Recording standin-baseline: the stand-in labels the 1,001 test messages (test_1001). Run once.

Stand-in: Qwen3-4B-Instruct-2507, Q4_K_M GGUF, llama-cpp-python 0.3.36, greedy, seed 42, 4 threads.
Writes examples/data/recorded/standin-baseline/answers.jsonl (resumable) and manifest.json.
Usage: python recordings/standin_baseline.py --gguf PATH [--limit N] [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

NAME = "standin-baseline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", help="path to the GGUF (or set STANDIN_GGUF)")
    ap.add_argument("--limit", type=int, help="first N messages of test_1001 only (smoke runs)")
    ap.add_argument("--out", help="output folder (default: examples/data/recorded/standin-baseline)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else c.RECORDED / NAME
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "answers.jsonl"
    rows = c.rl.read_rows("test")
    todo = [i for i in c.rl.read_sample("test_1001", args.limit) if i not in c.done_indices(jsonl)]

    c.rl.setup_runtime()
    llm, grammar = c.standin(c.gguf_path(args.gguf))
    with open(jsonl, "a") as f:
        for i in todo:
            label, text, logprob = c.standin_team(llm, grammar, rows[i][0])
            f.write(json.dumps({"index": i, "label": label, "team": text, "logprob": logprob}) + "\n")
            f.flush()

    if args.limit is None:
        c.write_manifest(
            out, "standin_baseline.py", c.STANDIN_REPO + " / " + c.STANDIN_FILE, c.STANDIN_REVISION,
            settings={"gguf_sha256": c.STANDIN_SHA256, "decoding": "greedy, temperature 0", "seed": 42,
                      "n_ctx": 2048, "n_threads": c.THREADS, "max_tokens": 16,
                      "constraint": "GBNF alternatives over the 77 labels", "chat_template": "embedded in the GGUF",
                      "sample": "test_1001", "messages": 1001, "logprob": "sum of answer token log-probabilities"},
            outputs=["answers.jsonl"], prompt_role="input: the router prompt, in the GGUF chat template",
        )
    print("done", len(c.done_indices(jsonl)), "answers")


if __name__ == "__main__":
    main()
