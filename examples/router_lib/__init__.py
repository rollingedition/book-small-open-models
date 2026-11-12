"""Shared code for the book's examples and recordings.

Labels, the router prompt, offline model loading, routing and metrics.
Callers set their own thread count. This module does not.
"""

import csv
import json
import os
import random
import tomllib
from pathlib import Path

# Set before transformers is imported, so the libraries read the pinned files offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent.parent  # examples/
BANKING = HERE / "data" / "banking77"
LABELS = json.loads((BANKING / "categories.json").read_text())
assert len(LABELS) == 77, len(LABELS)

PROMPT = (
    "Route this customer message from an online bank to one support team. "
    "Answer with the team name only.\n"
    "Teams: {teams}\n\n"
    "Message: {message}\n"
    "Team:"
)


def _pins():
    """Map each repo in models.toml to its pinned revision."""
    with open(HERE / "models.toml", "rb") as f:
        return {m["repo"]: m["revision"] for m in tomllib.load(f)["model"]}


def setup_runtime():
    """Fix the seeds, require deterministic algorithms and use float32."""
    random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)


def load_model(repo):
    """Return (tokenizer, causal model) for a repo pinned in models.toml. Reads offline."""
    revision = _pins()[repo]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, dtype=torch.float32, attn_implementation="eager"
    )
    return tokenizer, model.eval()


def load_classifier(adapter_dir):
    """Return (tokenizer, model) for a base model, its LoRA adapter and the 77-way head.

    The base repo is read from the adapter's adapter_config.json and its revision from models.toml.
    """
    from peft import PeftModel

    config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    repo = config["base_model_name_or_path"]
    revision = _pins()[repo]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        repo, revision=revision, num_labels=len(LABELS), dtype=torch.float32, attn_implementation="eager"
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, PeftModel.from_pretrained(base, adapter_dir).eval()


def router_prompt(message):
    """The router prompt for one message. Its text is the same in every live and recorded run."""
    return PROMPT.format(teams=", ".join(LABELS), message=message)


def build_trie(tokenizer):
    """Token trie over " " + label for every label.

    A node is {"children": {token_id: node}, "label": index or None}.
    """
    root = {"children": {}, "label": None}
    for index, name in enumerate(LABELS):
        node = root
        for token in tokenizer.encode(" " + name, add_special_tokens=False):
            node = node["children"].setdefault(token, {"children": {}, "label": None})
        assert node["label"] is None, f"label {name} ends where another label ends"
        node["label"] = index
    return root


def constrained_team(tokenizer, model, trie, message):
    """Greedy decoding that may emit only trie continuations, then end.

    Returns (label index, label text). The label index is None only if the decoding stopped early.
    """
    encoded = tokenizer(router_prompt(message), return_tensors="pt")
    start = encoded.input_ids.shape[1]
    eos = tokenizer.eos_token_id

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
            **encoded,
            max_new_tokens=32,
            do_sample=False,
            prefix_allowed_tokens_fn=allowed,
            pad_token_id=eos,
        )
    node, kept = trie, []
    for token in out[0, start:].tolist():
        if token == eos:
            break
        node = node["children"][token]
        kept.append(token)
    return node["label"], tokenizer.decode(kept, skip_special_tokens=True).strip()


def classify(model, tokenizer, messages, batch_size=16):
    """Return (label indices, softmax confidences) for the messages.

    One forward pass per batch. Ties break on the lower label index.
    """
    preds, confs = [], []
    with torch.no_grad():
        for start in range(0, len(messages), batch_size):
            batch = tokenizer(
                messages[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            probs = model(**batch).logits.softmax(dim=-1)
            pred = probs.argmax(dim=-1)
            preds += pred.tolist()
            confs += probs.gather(1, pred[:, None])[:, 0].tolist()
    return preds, confs


def handoff(confidence, threshold):
    """True when the confidence is below the threshold, so the message goes to a person.

    A confidence that isn't a number at or above the threshold (NaN included) is handed off too."""
    return not confidence >= threshold


def metrics(preds, golds, confs, threshold):
    """Accuracy, hand-off rate and accuracy on messages not handed off, with their counts."""
    n = len(golds)
    kept = [i for i in range(n) if not handoff(confs[i], threshold)]
    correct = sum(p == g for p, g in zip(preds, golds))
    correct_kept = sum(preds[i] == golds[i] for i in kept)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "handed_off": n - len(kept),
        "handoff_rate": (n - len(kept)) / n,
        "not_handed_off": len(kept),
        "correct_not_handed_off": correct_kept,
        "accuracy_not_handed_off": correct_kept / len(kept) if kept else None,
    }


def read_sample(name, n=None):
    """Row indices from samples.json: dev, test_1001, test_40 or test_200. Optional prefix n."""
    ids = json.loads((BANKING / "samples.json").read_text())[name]
    return ids[:n] if n else ids


def read_rows(split):
    """Return (text, label index) for every row of a split, in file order: "train" or "test"."""
    index = {name: i for i, name in enumerate(LABELS)}
    with open(BANKING / f"{split}.csv", newline="") as f:
        return [(row["text"], index[row["category"]]) for row in csv.DictReader(f)]


def pinned_revision(repo):
    """The revision a repo is pinned to in models.toml."""
    return _pins()[repo]


def training_rows(seed=0):
    """Training rows outside the dev sample, with their training order: a shuffle with the given seed.

    Returns (rows, order). rows are (text, label index); order holds positions in rows.
    """
    dev = set(read_sample("dev"))
    rows = [row for i, row in enumerate(read_rows("train")) if i not in dev]
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return rows, order


def new_classifier(repo):
    """Return (tokenizer, PeftModel): a base model with a fresh 77-way head and LoRA at the book's settings.

    Rank 8, lora_alpha 16, no dropout, targets q_proj and v_proj. task_type SEQ_CLS makes peft train the head with the adapter.
    Call setup_runtime() first, so the head and adapter start from the seed the recording used.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    revision = _pins()[repo]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    base = AutoModelForSequenceClassification.from_pretrained(
        repo, revision=revision, num_labels=len(LABELS), dtype=torch.float32, attn_implementation="eager"
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]
    )
    return tokenizer, get_peft_model(base, config)


def parameter_split(model):
    """Return (LoRA parameters, head parameters) counted from the trainable tensors.

    A tensor with "lora_" in its name is LoRA; every other trainable tensor is the head.
    """
    lora = head = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" in name:
                lora += p.numel()
            else:
                head += p.numel()
    return lora, head


def train_steps(model, tokenizer, rows, order, steps, batch_size=8, lr=1e-3):
    """Run AdamW on the trainable parameters for the given number of steps; return the loss at each step.

    Step k trains on rows[i] for i in order[k * batch_size:(k + 1) * batch_size]; rows are (text, label index).
    The model is left in eval mode.
    """
    import torch.nn.functional as F

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        batch = order[step * batch_size : (step + 1) * batch_size]
        enc = tokenizer([rows[i][0] for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=128)
        loss = F.cross_entropy(model(**enc).logits, torch.tensor([rows[i][1] for i in batch]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


def classify_by_length(model, tokenizer, texts, batch_size=16):
    """classify() on the texts sorted by token length, so each batch pads less; results are returned in input order.

    Returns (label indices, softmax confidences), in the order of texts.
    """
    lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
    by_length = sorted(range(len(texts)), key=lambda i: (lengths[i], i))
    sorted_preds, sorted_confs = classify(model, tokenizer, [texts[i] for i in by_length], batch_size)
    preds, confs = [0] * len(texts), [0.0] * len(texts)
    for k, i in enumerate(by_length):
        preds[i], confs[i] = sorted_preds[k], sorted_confs[k]
    return preds, confs


def choose_threshold(preds, golds, confs, target=0.90):
    """The lowest threshold on a 0.01 grid whose messages not handed off are at least `target` accurate.

    Returns None if no threshold on the grid meets the target.
    """
    def meets(threshold):
        r = metrics(preds, golds, confs, threshold)
        return r["not_handed_off"] > 0 and r["correct_not_handed_off"] * 10 >= target * 10 * r["not_handed_off"]

    grid = [round(k / 100, 2) for k in range(101)]
    return next((t for t in grid if meets(t)), None)


def score_test_sample(model, tokenizer, name, threshold):
    """metrics() for a fixed test sample from samples.json (for example "test_200") at one threshold."""
    test = read_rows("test")
    sample = read_sample(name)
    golds = [test[i][1] for i in sample]
    preds, confs = classify(model, tokenizer, [test[i][0] for i in sample])
    return metrics(preds, golds, confs, threshold)


def make_app(recorded_dir, threads=2):
    """A FastAPI app with one route, POST /route, serving the adapter in a recorded folder.

    First runs check_adapter and the base pin check, and raises RuntimeError on any mismatch, before any model loads.
    Then sets the process's intra-op thread count to threads, and loads the model, once per app. The
    threshold and the adapter's name come from the folder's manifest.json. The route computes
    handed_off with handoff(); team is None when it is True.
    """
    from fastapi import FastAPI
    from pydantic import BaseModel

    recorded_dir = Path(recorded_dir)
    manifest, s_ok, s_n, f_ok, f_n, pinned = check_adapter(recorded_dir)
    base_ok = pinned_revision(manifest["model"]) == manifest["model_version"]
    if not (pinned and base_ok and s_ok == s_n and f_ok == f_n):
        raise RuntimeError("adapter check failed: the app is not built")
    name = manifest["adapter"]
    version = name.rsplit("-", 1)[1]
    threshold = manifest["handoff_threshold"]
    torch.set_num_threads(threads)
    tokenizer, model = load_classifier(recorded_dir / "adapter")

    class RouteIn(BaseModel):
        message: str

    app = FastAPI()

    @app.post("/route")
    def route(body: RouteIn):
        preds, confs = classify(model, tokenizer, [body.message])
        flag = handoff(confs[0], threshold)
        return {
            "team": None if flag else LABELS[preds[0]],
            "confidence": confs[0],
            "handed_off": flag,
            "adapter": name,
            "version": version,
        }

    return app


def int8_classifier(adapter_dir):
    """Return (tokenizer, model): the adapter merged into its base, every backbone Linear layer quantized, the score head in float32.

    Quantization is torchao's Int8WeightOnlyConfig: per-row int8 weights, float32 activations. Call setup_runtime() first.
    """
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    tokenizer, peft_model = load_classifier(adapter_dir)
    model = peft_model.merge_and_unload()
    quantize_(
        model,
        Int8WeightOnlyConfig(),
        filter_fn=lambda module, name: isinstance(module, torch.nn.Linear) and name != "score",
    )
    return tokenizer, model.eval()


def backbone_linears(model):
    """(name, layer) for every nn.Linear except the score head, in module order."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and name != "score"]


def is_int8(layer):
    """True when the layer's weight is a torchao quantized tensor rather than a float tensor."""
    return type(layer.weight).__module__.startswith("torchao")


def int8_classifier_all_linears(adapter_dir):
    """Return (tokenizer, model) like int8_classifier, but with the score head quantized too.

    Used only to count what quantizing the head changes. Call setup_runtime() first.
    """
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    tokenizer, peft_model = load_classifier(adapter_dir)
    model = peft_model.merge_and_unload()
    quantize_(model, Int8WeightOnlyConfig(), filter_fn=lambda module, name: isinstance(module, torch.nn.Linear))
    return tokenizer, model.eval()


def print_copy_report(name, preds, confs, golds, threshold):
    """Print one copy's accuracy, hand-offs and accuracy on the messages not handed off, as counts and rates."""
    r = metrics(preds, golds, confs, threshold)
    print(f"{name} accuracy on test_200: {r['correct']}/{r['n']} ({r['accuracy']:.1%})")
    print(f"{name} handed off: {r['handed_off']}/{r['n']} ({r['handoff_rate']:.1%})")
    kept = r["not_handed_off"]
    if kept:
        print(
            f"{name} accuracy on messages not handed off: "
            f"{r['correct_not_handed_off']}/{kept} ({r['accuracy_not_handed_off']:.1%})"
        )
    else:
        print(f"{name} accuracy on messages not handed off: not defined (0 messages not handed off)")


def count_changes(preds_a, confs_a, preds_b, confs_b, threshold):
    """(predictions that differ, hand-off decisions that differ) between two copies on the same messages."""
    changed = sum(a != b for a, b in zip(preds_a, preds_b))
    handoffs = sum(handoff(x, threshold) != handoff(y, threshold) for x, y in zip(confs_a, confs_b))
    return changed, handoffs


def best_threshold(preds, golds, confs):
    """The threshold on the 0.01 grid with the highest accuracy on messages not handed off (the lowest on ties).

    Used when choose_threshold finds no threshold at its target. Thresholds handing off every message are skipped.
    """
    best, best_accuracy = None, -1.0
    for t in [round(k / 100, 2) for k in range(101)]:
        r = metrics(preds, golds, confs, t)
        if r["not_handed_off"] and r["accuracy_not_handed_off"] > best_accuracy:
            best, best_accuracy = t, r["accuracy_not_handed_off"]
    return best


def dev_threshold(model, tokenizer, target=0.90, n=None):
    """(threshold, reached) for a classifier on the dev split: choose_threshold at the target, else best_threshold.

    n: use only the first n dev messages (the time budget); None uses the whole dev sample.
    """
    train = read_rows("train")
    dev = read_sample("dev", n)
    golds = [train[i][1] for i in dev]
    preds, confs = classify_by_length(model, tokenizer, [train[i][0] for i in dev])
    chosen = choose_threshold(preds, golds, confs, target)
    if chosen is not None:
        return chosen, True
    return best_threshold(preds, golds, confs), False


def standin_answers(name="standin-baseline"):
    """{test row index: label} from a recording's answers.jsonl, for example the stand-in's test answers."""
    path = HERE / "data" / "recorded" / name / "answers.jsonl"
    return {row["index"]: row["label"] for row in map(json.loads, path.read_text().splitlines())}


def standin_rows(name="standin-baseline"):
    """{test row index: {"team", "label", "logprob"}} from a recording's answers.jsonl, with each answer's log-probability."""
    path = HERE / "data" / "recorded" / name / "answers.jsonl"
    return {row["index"]: row for row in map(json.loads, path.read_text().splitlines())}


def sha256_file(path):
    """The SHA-256 hex digest of a file, read in 1 MiB blocks."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_pins():
    """The SHA-256 of each recorded adapter's manifest.json, pinned with the code in examples/adapter-pins.json."""
    return json.loads((HERE / "adapter-pins.json").read_text())


def check_adapter(folder):
    """(manifest, settings matching, settings checked, files matching, files checked, manifest matches pin).

    Settings: r, lora_alpha and target_modules in adapter_config.json against the manifest's lora settings.
    Files: the SHA-256 of each output in the manifest's output_sha256, against the digest recorded there.
    Manifest: the SHA-256 of manifest.json itself, against the digest pinned for its adapter name. The pin covers
    every field the manifest holds, including handoff_threshold, model_version and the output digests."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    config = json.loads((folder / "adapter" / "adapter_config.json").read_text())
    lora = manifest["settings"]["lora"]
    settings = [
        config["r"] == lora["r"],
        config["lora_alpha"] == lora["alpha"],
        sorted(config["target_modules"]) == sorted(lora["targets"]),
    ]
    files = [sha256_file(folder / name) == digest for name, digest in manifest["output_sha256"].items()]
    pin = adapter_pins().get(manifest.get("adapter"), {}).get("manifest_sha256")
    pinned = pin is not None and sha256_file(folder / "manifest.json") == pin
    return manifest, sum(settings), len(settings), sum(files), len(files), pinned


def edited_copy(folder, dest, path, key, value):
    """Copy a recorded adapter folder to dest, then set one JSON key in the file at path (relative to the folder)."""
    import shutil

    shutil.copytree(folder, dest)
    data = json.loads((dest / path).read_text())
    data[key] = value
    (dest / path).write_text(json.dumps(data, indent=2))
    return dest


def refused_by_make_app(folder):
    """True when make_app raises RuntimeError on the folder, before it loads a model. Call it only on a folder that fails."""
    try:
        make_app(folder)
    except RuntimeError:
        return True
    return False


# The hand-off watch. The baseline is the dev split's hand-off rate at the adapter's threshold, computed by dev_handed_off.
# DRIFT_POINTS is the book's choice, not a sourced number: a week more than this many points from the baseline drifts.
DRIFT_POINTS = 10.0
WEEK_SIZE = 25


def dev_handed_off(model, tokenizer, threshold):
    """(messages handed off, dev messages) at the threshold: the dev split's hand-off count, the watch's baseline."""
    train = read_rows("train")
    dev = read_sample("dev")
    _, confs = classify_by_length(model, tokenizer, [train[i][0] for i in dev])
    return sum(handoff(c, threshold) for c in confs), len(dev)


def watch_weeks(model, tokenizer, threshold, simulated=True):
    """Fixed weekly windows of test_1001 for the hand-off watch: [(row indices, predictions, confidences)] per week.

    Weeks 1 and 2 are the first 50 sample messages in order. With simulated=True, weeks 3 and 4 are a replay, not
    new traffic: the sample's messages 51 to 100, sorted by router confidence (lowest first, ties by position), split
    into two weeks of 25. Only the messages routed are classified, once each, with classify_by_length."""
    rows = read_sample("test_1001")[: 4 * WEEK_SIZE if simulated else 2 * WEEK_SIZE]
    test = read_rows("test")
    preds, confs = classify_by_length(model, tokenizer, [test[i][0] for i in rows])
    order = [list(range(0, WEEK_SIZE)), list(range(WEEK_SIZE, 2 * WEEK_SIZE))]
    if simulated:
        replay = sorted(range(2 * WEEK_SIZE, 4 * WEEK_SIZE), key=lambda k: (confs[k], k))
        order += [replay[:WEEK_SIZE], replay[WEEK_SIZE:]]
    return [([rows[k] for k in ks], [preds[k] for k in ks], [confs[k] for k in ks]) for ks in order]


def chat_prompt(tokenizer, message):
    """The router prompt in the model's chat template, with its default system prompt. Tokenize with add_special_tokens=False."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": router_prompt(message)}], tokenize=False, add_generation_prompt=True
    )


def constrained_team_text(tokenizer, model, trie, text, add_special_tokens=True):
    """constrained_team for a prompt given as text (for example from chat_prompt). Returns (label index, label text)."""
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    start = encoded.input_ids.shape[1]
    eos = tokenizer.eos_token_id

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


def label_token_table(tokenizer):
    """(token lists, padded tensor) for " " + each label; the tensor rows are padded with EOS to the longest label."""
    label_tokens = [tokenizer.encode(" " + name, add_special_tokens=False) for name in LABELS]
    width = max(len(t) for t in label_tokens)
    padded = torch.full((len(label_tokens), width), tokenizer.eos_token_id)
    for k, t in enumerate(label_tokens):
        padded[k, : len(t)] = torch.tensor(t)
    return label_tokens, padded


def label_scores(tokenizer, model, table, message):
    """Summed log-probability of each team's tokens after the router prompt, one score per label index.

    The prompt is run once; its cache is copied for the 77 teams, which are scored in one batch.
    """
    label_tokens, padded = table
    prompt = tokenizer(router_prompt(message)).input_ids
    with torch.no_grad():
        first = model(input_ids=torch.tensor([prompt]), use_cache=True)
        start = first.logits[0, -1].log_softmax(-1)
        cache = first.past_key_values
        cache.batch_repeat_interleave(len(label_tokens))
        rest = model(input_ids=padded, past_key_values=cache, use_cache=True).logits.log_softmax(-1)
    scores = []
    for k, t in enumerate(label_tokens):
        total = start[t[0]].item()
        for j in range(1, len(t)):
            total += rest[k, j - 1, t[j]].item()
        scores.append(total)
    return scores


def router_prompt_tokens(messages):
    """Mean router-prompt token count per message, rounded to a whole token, with the 135M model's tokenizer. Counted live; no model runs."""
    repo = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=_pins()[repo])
    counts = [len(tokenizer.encode(router_prompt(message))) for message in messages]
    return round(sum(counts) / len(counts))


def bill_inputs():
    """The reader's inputs for the bill and the break-even. Example values, labelled as such; none is a published price."""
    return {
        "messages_per_month": 100_000,
        "price_per_million_input_tokens": 1.00,
        "price_per_million_output_tokens": 4.00,
        "output_tokens_per_message": 6,
        "machine_cost_per_hour": 0.40,
        "hours_per_month": 730,
        "router_messages_per_second": 20,
        "utilisation": 0.5,
    }


def recorded_rates():
    """Messages per second as recorded on the machine named in the throughput manifest: rates.json."""
    return json.loads((HERE / "data" / "recorded" / "throughput" / "rates.json").read_text())


def bill_inputs_recorded():
    """bill_inputs() with the router's rate taken from the recording: the classifier at one message per call, the
    like-for-like rate against the stand-in's one message per call. The other inputs stay example values."""
    inputs = bill_inputs()
    inputs["router_messages_per_second"] = recorded_rates()["classifier_fp32_one_message_per_call"]
    return inputs


def compare_router_and_standin(sample):
    """The router (banking77-lora-v1, live, classify_by_length) and the stand-in (recorded answers) on one committed
    sample. Returns counts: messages, correct for each, handed off, and the router's wrong answers sent as answers."""
    adapter = HERE / "data" / "recorded" / "lora-full"
    threshold = json.loads((adapter / "manifest.json").read_text())["handoff_threshold"]
    answers = standin_rows()
    test = read_rows("test")
    rows = read_sample(sample)
    golds = [test[i][1] for i in rows]
    standin_right = [answers[i]["label"] == g for i, g in zip(rows, golds)]
    tokenizer, model = load_classifier(adapter / "adapter")
    preds, confs = classify_by_length(model, tokenizer, [test[i][0] for i in rows])
    router_right = [p == g for p, g in zip(preds, golds)]
    handed_off = [handoff(c, threshold) for c in confs]
    return {
        "messages": len(rows),
        "router_correct": sum(router_right),
        "standin_correct": sum(standin_right),
        "handed_off": sum(handed_off),
        "standin_right_router_wrong": sum(s and not r for s, r in zip(standin_right, router_right)),
        "of_those_handed_off": sum(s and not r and h for s, r, h in zip(standin_right, router_right, handed_off)),
        "router_wrong_sent": sum(not r and not h for r, h in zip(router_right, handed_off)),
        "router_right_standin_wrong": sum(r and not s for r, s in zip(router_right, standin_right)),
    }
