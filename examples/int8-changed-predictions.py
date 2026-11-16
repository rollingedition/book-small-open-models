"""Score test_200 with the float32 adapter, its int8 copy and the copy with the head in int8, at the recorded threshold.
The prose's point rests on the line that counts predictions changed by int8, read with its hand-off count.
Prints counts and rates only: accuracy, hand-offs, and how many predictions and hand-off decisions change."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(2)

import router_lib  # noqa: E402

router_lib.setup_runtime()
RECORDED = router_lib.HERE / "data" / "recorded" / "lora-full"
threshold = json.loads((RECORDED / "manifest.json").read_text())["handoff_threshold"]  # chosen on dev in chapter 5, stored in the manifest
test = router_lib.read_rows("test")  # (text, label index) pairs
sample = router_lib.read_sample("test_200")  # fixed row indices, so every copy scores the same messages
golds = [test[i][1] for i in sample]  # index 1 of each row is its gold team
texts = [test[i][0] for i in sample]

tokenizer, float_model = router_lib.load_classifier(RECORDED / "adapter")  # the adapter as chapter 5 loaded it, unmerged
float_preds, float_confs = router_lib.classify_by_length(float_model, tokenizer, texts)  # results in the order of texts
_, int8_model = router_lib.int8_classifier(RECORDED / "adapter")  # merged first, then backbone quantized, head float32
int8_preds, int8_confs = router_lib.classify_by_length(int8_model, tokenizer, texts)
_, head_model = router_lib.int8_classifier_all_linears(RECORDED / "adapter")  # as above, with the score head quantized too
head_preds, head_confs = router_lib.classify_by_length(head_model, tokenizer, texts)

print(f"handoff threshold from manifest: {threshold:.2f}")
print(f"messages on test_200: {len(texts)}")
router_lib.print_copy_report("float32", float_preds, float_confs, golds, threshold)
router_lib.print_copy_report("int8", int8_preds, int8_confs, golds, threshold)
changed, handoffs = router_lib.count_changes(float_preds, float_confs, int8_preds, int8_confs, threshold)  # same threshold for both copies
print(f"predictions changed by int8: {changed}/{len(texts)} ({changed / len(texts):.1%})")
print(f"hand-off decisions changed by int8: {handoffs}/{len(texts)}")
changed_head, _ = router_lib.count_changes(float_preds, float_confs, head_preds, head_confs, threshold)  # hand-off count not printed here
print(f"predictions changed with the head in int8 too: {changed_head}/{len(texts)}")
