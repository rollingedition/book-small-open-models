"""Helpers shared by the recordings: pins, hashes, the stand-in, LoRA training and manifests.

Recordings run once, on the machine named in recordings/README.md. Their outputs are read offline by the examples.
"""

import hashlib
import json
import os
import random
import sys
import tomllib
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
RECORDED = EXAMPLES / "data" / "recorded"
sys.path.insert(0, str(EXAMPLES))

import router_lib as rl  # noqa: E402

BANKING_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
STANDIN_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
STANDIN_REVISION = "a06e946bb6b655725eafa393f4a9745d460374c9"
STANDIN_FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
STANDIN_SHA256 = "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"
BASE_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
THREADS = 4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prompt_sha256():
    """SHA-256 of the router prompt with the message left as a placeholder."""
    return hashlib.sha256(rl.router_prompt("{message}").encode()).hexdigest()


def gguf_path(arg=None):
    """The stand-in's GGUF file, from --gguf or the STANDIN_GGUF variable. Checked against the pin."""
    path = Path(arg or os.environ.get("STANDIN_GGUF", ""))
    if not path.is_file():
        sys.exit("Set STANDIN_GGUF or --gguf to the GGUF file.")
    if sha256_file(path) != STANDIN_SHA256:
        sys.exit("The GGUF does not match the pinned SHA-256.")
    return path


def pinned_revision(repo):
    with open(EXAMPLES / "models.toml", "rb") as f:
        return {m["repo"]: m["revision"] for m in tomllib.load(f)["model"]}[repo]


def done_indices(jsonl):
    """Indices already written to a resumable jsonl file."""
    if not jsonl.exists():
        return set()
    return {json.loads(line)["index"] for line in open(jsonl)}


def write_manifest(out_dir, script, model, model_version, settings, outputs, prompt_role, extra=None):
    """Write manifest.json: script, model, versions, prompt hash, date and each output's SHA-256."""
    manifest = {
        "script": f"recordings/{script}",
        "model": model,
        "model_version": model_version,
        "prompt_sha256": prompt_sha256(),
        "prompt_role": prompt_role,
        "created": date.today().isoformat(),
        "recipe_sha256": sha256_file(HERE / script),
        "data_commit": BANKING_COMMIT,
        "settings": settings,
        "outputs": list(outputs),
        "output_sha256": {rel: sha256_file(out_dir / rel) for rel in outputs},
    }
    manifest.update(extra or {})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def standin(gguf):
    """Load the stand-in with llama-cpp-python at the pinned settings."""
    from llama_cpp import Llama, LlamaGrammar

    grammar_text = "root ::= " + " | ".join(f'"{name}"' for name in rl.LABELS)
    llm = Llama(model_path=str(gguf), n_ctx=2048, n_threads=THREADS, n_threads_batch=THREADS, seed=42, logits_all=True, verbose=False)
    return llm, LlamaGrammar.from_string(grammar_text)


def standin_team(llm, grammar, message):
    """Greedy team label from the stand-in, limited to the 77 labels. Returns (label index, text, logprob)."""
    reply = llm.create_chat_completion(
        messages=[{"role": "user", "content": rl.router_prompt(message)}],
        temperature=0.0,
        max_tokens=16,
        seed=42,
        grammar=grammar,
        logprobs=True,
        top_logprobs=1,
    )
    choice = reply["choices"][0]
    text = choice["message"]["content"].strip()
    logprob = sum(float(tok["logprob"]) for tok in choice["logprobs"]["content"])
    return rl.LABELS.index(text), text, round(logprob, 4)


def train_lora(texts, labels, out_dir, limit=None):
    """One epoch of LoRA r=8 on q_proj and v_proj, with the 77-way head trainable. Saves the adapter.

    AdamW, lr 1e-3, batch 8, float32, seed 0, rows in a fixed shuffled order. Returns the per-step loss.
    """
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo = BASE_REPO
    revision = pinned_revision(repo)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        repo, revision=revision, num_labels=len(rl.LABELS), dtype=torch.float32, attn_implementation="eager"
    )
    base.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(
        base,
        LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]),
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    order = list(range(len(texts)))
    random.Random(0).shuffle(order)
    model.train()
    losses = []
    for step, start in enumerate(range(0, len(order), 8)):
        batch = order[start : start + 8]
        enc = tok([texts[i] for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=128)
        loss = F.cross_entropy(model(**enc).logits, torch.tensor([labels[i] for i in batch]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 4))
        if limit and step + 1 >= limit:
            break
    model.eval()
    model.save_pretrained(out_dir)
    return losses
