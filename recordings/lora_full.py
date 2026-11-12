#!/usr/bin/env python3
"""Recording lora-full: the recorded full fine-tune, banking77-lora-v1. Run once, 4 threads.

SmolLM2-135M-Instruct, LoRA r=8, alpha 16, q_proj and v_proj, 77-way head trainable. AdamW lr 1e-3, batch 8,
seed 0, one epoch over the training rows not in dev (about 9,000 messages).
Writes examples/data/recorded/lora-full/adapter/ (safetensors) and steps.jsonl, then manifest.json.
Usage: python recordings/lora_full.py [--limit STEPS] [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

NAME = "lora-full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="stop after N steps (smoke runs)")
    ap.add_argument("--out", help="output folder (default: examples/data/recorded/lora-full)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else c.RECORDED / NAME
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    rows = c.rl.read_rows("train")
    dev = set(c.rl.read_sample("dev"))
    train = [i for i in range(len(rows)) if i not in dev]

    c.rl.setup_runtime()
    import torch
    torch.set_num_threads(c.THREADS)
    losses = c.train_lora([rows[i][0] for i in train], [rows[i][1] for i in train], out / "adapter", args.limit)
    with open(out / "steps.jsonl", "w") as f:
        for step, loss in enumerate(losses, 1):
            f.write(json.dumps({"step": step, "loss": loss}) + "\n")

    # The hand-off threshold, chosen on the dev split only (plan, "Decided by the managing editor"): the lowest
    # threshold on a 0.01 grid at which accuracy on the messages not handed off is at least 0.90.
    tok, model = c.rl.load_classifier(out / "adapter")
    dev_idx = sorted(dev)
    preds, confs = c.rl.classify(model, tok, [rows[i][0] for i in dev_idx])
    golds = [rows[i][1] for i in dev_idx]
    threshold = None
    for step in range(0, 100):
        t = step / 100
        kept = [p == g for p, g, q in zip(preds, golds, confs) if not c.rl.handoff(q, t)]
        if kept and sum(kept) / len(kept) >= 0.90:
            threshold = t
            break
    print("dev threshold", threshold)

    if args.limit is None:
        c.write_manifest(
            out, "lora_full.py", c.BASE_REPO, c.pinned_revision(c.BASE_REPO),
            settings={"lora": {"r": 8, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "v_proj"], "head": "trainable"},
                      "optimizer": "AdamW lr 1e-3", "batch_size": 8, "epochs": 1, "seed": 0, "dtype": "float32",
                      "max_length": 128, "n_threads": c.THREADS, "train_rows": len(train), "dev_excluded": True,
                      "steps": len(losses), "data": "train.csv rows not in dev"},
            outputs=["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "steps.jsonl"],
            prompt_role="not used: the classifier reads the raw message text",
            extra={"adapter": "banking77-lora-v1", "handoff_threshold": threshold,
                   "threshold_rule": "lowest t on a 0.01 grid with accuracy >= 0.90 on dev messages not handed off"},
        )
    print("done", len(losses), "steps")


if __name__ == "__main__":
    main()
