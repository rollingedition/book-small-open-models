"""Draw the length in tokens of each test_200 message as a histogram, with the 64-token cut and the 128-token cut that classify() applies as lines. The printed counts are the ones truncation-count prints, recomputed the same way; only the tokenizer loads."""

from lib.figures import ACCENT, INK, new_figure, save

import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402

torch.set_num_threads(2)

from transformers import AutoTokenizer  # noqa: E402

import router_lib  # noqa: E402

router_lib.setup_runtime()
RECORDED = router_lib.HERE / "data" / "recorded" / "lora-full"
repo = json.loads((RECORDED / "adapter" / "adapter_config.json").read_text())["base_model_name_or_path"]  # the adapter's base model
tokenizer = AutoTokenizer.from_pretrained(repo, revision=router_lib.pinned_revision(repo))  # the pinned revision the model uses

test = router_lib.read_rows("test")
sample = router_lib.read_sample("test_200")
lengths = [len(tokenizer(test[i][0]).input_ids) for i in sample]  # full length in tokens, before any cut
n = len(lengths)

print(f"messages measured: {n}")
print(f"median length in tokens: {statistics.median(lengths):g}")
print(f"longest message in tokens: {max(lengths)}")
print(f"messages over 64 tokens: {sum(x > 64 for x in lengths)}/{n}")
print(f"messages over 128 tokens where classify() cuts: {sum(x > 128 for x in lengths)}/{n}")

fig, ax = new_figure()
ax.hist(lengths, bins=range(0, max(lengths) + 2), color=ACCENT, label="test_200 messages")  # one bin per token count
ax.axvline(64, color=INK, label="64-token cut")
ax.axvline(128, color=INK, linestyle="--", label="128-token cut, classify()")
ax.set_xlim(0, 136)
ax.set_xlabel("message length, in tokens")
ax.set_ylabel("messages")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False)
save(fig, "message-token-lengths")
