#!/usr/bin/env python3
"""Recording prompting-eval: constrained team decoding on the 200 test messages (test_200), three prompt variants. Run once.

Each variant decodes exactly as its live example does:
  135m-raw   SmolLM2-135M-Instruct, router prompt raw                   (constrained-decode-135m.py)
  360m-raw   SmolLM2-360M-Instruct, router prompt raw                   (constrained-decode-360m.py)
  135m-chat  SmolLM2-135M-Instruct, router prompt in its chat template  (constrained-decode-135m-chat.py)
Writes examples/data/recorded/prompting-eval/answers.jsonl (resumable: one line per variant and index) and manifest.json.
Usage: python recordings/prompting_eval.py [--limit N] [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

NAME = "prompting-eval"
SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MEDIUM = "HuggingFaceTB/SmolLM2-360M-Instruct"
VARIANTS = {"135m-raw": SMALL, "360m-raw": MEDIUM, "135m-chat": SMALL}


def chat_team(tokenizer, model, trie, message):
    """Greedy team decoding from the chat-formatted router prompt. Returns (label index, label text)."""
    eos = tokenizer.eos_token_id
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": c.rl.router_prompt(message)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    start = encoded.input_ids.shape[1]

    def allowed(_batch, ids):
        node = trie
        for token in ids[start:].tolist():
            if token == eos:
                break
            node = node["children"][token]
        options = list(node["children"])
        if node["label"] is not None:
            options.append(eos)
        return options or [eos]

    with torch.no_grad():
        out = model.generate(
            **encoded, max_new_tokens=32, do_sample=False, prefix_allowed_tokens_fn=allowed, pad_token_id=eos
        )
    node, kept = trie, []
    for token in out[0, start:].tolist():
        if token == eos:
            break
        node = node["children"][token]
        kept.append(token)
    return node["label"], tokenizer.decode(kept, skip_special_tokens=True).strip()


def done_pairs(jsonl):
    """(variant, index) pairs already written to answers.jsonl."""
    if not jsonl.exists():
        return set()
    with open(jsonl) as f:
        return {(row["variant"], row["index"]) for row in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="first N messages of test_200 only (smoke runs)")
    ap.add_argument("--out", help="output folder (default: examples/data/recorded/prompting-eval)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else c.RECORDED / NAME
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "answers.jsonl"
    rows = c.rl.read_rows("test")
    sample = c.rl.read_sample("test_200", args.limit)

    torch.set_num_threads(c.THREADS)
    c.rl.setup_runtime()
    done = done_pairs(jsonl)
    with open(jsonl, "a") as f:
        for variant, repo in VARIANTS.items():
            todo = [i for i in sample if (variant, i) not in done]
            if not todo:
                continue
            tokenizer, model = c.rl.load_model(repo)
            trie = c.rl.build_trie(tokenizer)
            for i in todo:
                if variant == "135m-chat":
                    label, team = chat_team(tokenizer, model, trie, rows[i][0])
                else:
                    label, team = c.rl.constrained_team(tokenizer, model, trie, rows[i][0])
                f.write(json.dumps({"variant": variant, "index": i, "label": label, "team": team}) + "\n")
                f.flush()

    if args.limit is None:
        raw = "router prompt, raw, add_special_tokens default"
        c.write_manifest(
            out, "prompting_eval.py", SMALL + ", " + MEDIUM,
            {SMALL: c.pinned_revision(SMALL), MEDIUM: c.pinned_revision(MEDIUM)},
            settings={
                "variants": {
                    "135m-raw": {"model": SMALL, "prompt": raw, "decoding": "greedy, max_new_tokens 32, prefix-constrained to the 77 labels"},
                    "360m-raw": {"model": MEDIUM, "prompt": raw, "decoding": "greedy, max_new_tokens 32, prefix-constrained to the 77 labels"},
                    "135m-chat": {"model": SMALL, "prompt": "router prompt in the chat template, with the model's default system prompt",
                                  "decoding": "greedy, max_new_tokens 32, prefix-constrained to the 77 labels, add_special_tokens False"},
                },
                "dtype": "float32", "attn_implementation": "eager", "threads": c.THREADS,
                "sample": "test_200", "messages": 200, "label_text": "team, one of the 77 labels",
            },
            outputs=["answers.jsonl"],
            prompt_role="router prompt: raw for 135m-raw and 360m-raw; in the SmolLM2 chat template for 135m-chat",
        )
    print("done", len(done_pairs(jsonl)), "answers")


if __name__ == "__main__":
    main()
