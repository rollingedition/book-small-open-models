#!/usr/bin/env python3
"""Recording throughput: messages per second on test_200, 4 threads, on the machine named in the manifest. Run once.

Compares the stand-in (Qwen3-4B GGUF), the recorded lora-full classifier in fp32 and the same classifier after
`banking77-lora-v1-int8` from `router_lib.int8_classifier` (merged LoRA, torchao Int8WeightOnlyConfig on the backbone, score head in float32). Ratios and, for cost arithmetic, rates to two significant figures, with the machine named.
Needs examples/data/recorded/lora-full/adapter (run lora_full.py first).
Writes examples/data/recorded/throughput/ratios.json, then manifest.json.
Usage: python recordings/throughput.py --gguf PATH [--machine TEXT] [--limit N] [--adapter DIR] [--out DIR]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

NAME = "throughput"
MACHINE = "4 vCPU, 15 GiB cloud sandbox"


def rate(fn, messages):
    """Messages per second for fn over the messages, after one warm-up call."""
    fn(messages[:1])
    start = time.perf_counter()
    fn(messages)
    return len(messages) / (time.perf_counter() - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", help="path to the GGUF (or set STANDIN_GGUF)")
    ap.add_argument("--machine", default=MACHINE, help="description of the machine, written to the manifest")
    ap.add_argument("--limit", type=int, help="first N messages of test_200 only (smoke runs)")
    ap.add_argument("--adapter", help="classifier adapter folder (default: recorded lora-full adapter)")
    ap.add_argument("--out", help="output folder (default: examples/data/recorded/throughput)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else c.RECORDED / NAME
    out.mkdir(parents=True, exist_ok=True)
    rows = c.rl.read_rows("test")
    messages = [rows[i][0] for i in c.rl.read_sample("test_200", args.limit)]
    adapter = Path(args.adapter) if args.adapter else c.RECORDED / "lora-full" / "adapter"

    c.rl.setup_runtime()
    import torch
    torch.set_num_threads(c.THREADS)

    llm, grammar = c.standin(c.gguf_path(args.gguf))
    standin = rate(lambda ms: [c.standin_team(llm, grammar, m) for m in ms], messages)
    del llm

    tok, model = c.rl.load_classifier(adapter)
    model = model.merge_and_unload().eval()
    fp32 = rate(lambda ms: c.rl.classify(model, tok, ms), messages)
    preds_fp32, _ = c.rl.classify(model, tok, messages)
    _, qmodel = c.rl.int8_classifier(adapter)  # the chapter 7 copy, exactly as the examples build it
    int8 = rate(lambda ms: c.rl.classify(qmodel, tok, ms), messages)
    preds_int8, _ = c.rl.classify(qmodel, tok, messages)
    # One message per call, as the stand-in answers: the like-for-like comparison.
    fp32_one = rate(lambda ms: [c.rl.classify(model, tok, [m]) for m in ms], messages)

    ratios = {
        "messages": len(messages),
        "classifier_fp32_over_standin": round(fp32 / standin, 1),
        "classifier_int8_over_standin": round(int8 / standin, 1),
        "classifier_int8_over_fp32": round(int8 / fp32, 1),
        "agreement_fp32_int8": sum(a == b for a, b in zip(preds_fp32, preds_int8)),
    }
    ratios["classifier_fp32_one_at_a_time_over_standin"] = round(fp32_one / standin, 1)
    (out / "ratios.json").write_text(json.dumps(ratios, indent=2) + "\n")

    def sig2(x):  # two significant figures: a measured rate on this machine, not a guarantee
        return float(f"{x:.2g}")

    rates = {
        "machine": args.machine,
        "unit": "messages per second, on test_200, after one warm-up call",
        "standin_one_message_per_call": sig2(standin),
        "classifier_fp32_batches_of_16": sig2(fp32),
        "classifier_int8_batches_of_16": sig2(int8),
        "classifier_fp32_one_message_per_call": sig2(fp32_one),
    }
    (out / "rates.json").write_text(json.dumps(rates, indent=2) + "\n")

    if args.limit is None:
        c.write_manifest(
            out, "throughput.py", c.STANDIN_REPO + " / " + c.STANDIN_FILE + "; " + c.BASE_REPO,
            c.STANDIN_REVISION,
            settings={"sample": "test_200", "n_threads": c.THREADS, "classifier_adapter": "banking77-lora-v1",
                      "int8": "banking77-lora-v1-int8: merged LoRA, torchao 0.18.0 Int8WeightOnlyConfig on backbone Linear layers, score head float32", "batch_size": 16,
                      "gguf_sha256": c.STANDIN_SHA256, "output": "ratios, and rates to two significant figures with the machine named (proposal, risk 5)"},
            outputs=["ratios.json", "rates.json"], prompt_role="input to the stand-in only; the classifier reads the raw message",
            extra={"machine": args.machine},
        )
    print("done", json.dumps(ratios))


if __name__ == "__main__":
    main()
