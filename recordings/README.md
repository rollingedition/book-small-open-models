# Recordings

Each recording runs once, on the recording machine (4 vCPU, 15 GiB), and writes `examples/data/recorded/{name}/`
with its outputs and a `manifest.json` (script, model, model version, prompt hash, date, output SHA-256s, settings).
The examples read those outputs offline. Nothing here runs in an example.

Setup: a Python 3.12 environment with `pip install -r recordings/requirements.txt`. The stand-in needs the GGUF file
from `unsloth/Qwen3-4B-Instruct-2507-GGUF` at revision `a06e946bb6b655725eafa393f4a9745d460374c9`, file
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`. Pass its path with `STANDIN_GGUF` or `--gguf`. The script checks its SHA-256
against the pin. The small models are fetched offline by revision, as in `examples/models.toml`.

Run from the book's root, in this order (`throughput` needs `lora_full`; `distill_student` needs `teacher_labels`).

| Script | Records | Machine | Command | Estimate |
| --- | --- | --- | --- | --- |
| `standin_baseline.py` | `standin-baseline`: stand-in labels for `test_1001` | recording machine | `STANDIN_GGUF=/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf python recordings/standin_baseline.py` | about 25 min |
| `lora_full.py` | `lora-full`: `banking77-lora-v1` adapter | recording machine | `python recordings/lora_full.py` | about 15 min |
| `teacher_labels.py` | `teacher-labels`: stand-in labels for 2,000 training rows | recording machine | `STANDIN_GGUF=/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf python recordings/teacher_labels.py` | about 50 min |
| `distill_student.py` | `distill-student`: `banking77-distilled-v1` adapter | recording machine | `python recordings/distill_student.py` | about 4 min |
| `throughput.py` | `throughput`: ratios on `test_200` | recording machine | `STANDIN_GGUF=/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf python recordings/throughput.py --machine "4 vCPU, 15 GiB cloud sandbox"` | about 15 min |
| `prompting_eval.py` | `prompting-eval`: constrained team decoding on `test_200` for three prompts (`135m-raw`, `360m-raw`, `135m-chat`) | recording machine | `python recordings/prompting_eval.py` | about 3 h (provisional: the 3-message smoke run took about 5 min on two shared vCPUs) |

Estimates are the planner's provisional figures. Long runs resume: a rerun skips the rows already in the output file.
Add `--limit N` to any script for a short check; a limited run writes no manifest.

**Licences of the adapters.** Each training script leaves PEFT's boilerplate `adapter/README.md`; after a re-recording, replace it with the short model card the other adapters carry (licence `apache-2.0`, base model, a pointer to `../manifest.json` and `../../NOTICE.md`). `examples/data/recorded/NOTICE.md` and `LICENSE-APACHE-2.0.txt` cover every adapter (Apache-2.0 §4(a) and (b)).
