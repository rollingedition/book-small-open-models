# Notice for the recorded adapters

The adapters under `lora-full/adapter/`, `distill-student/adapter/` and `gold-same-rows/adapter/` are LoRA weights and a classification head trained on top of **SmolLM2-135M-Instruct** (`HuggingFaceTB/SmolLM2-135M-Instruct`, at the revision pinned in `examples/models.toml`), which is licensed under the Apache License, Version 2.0. They are derivative works of that model, modified by training: they are not the model's original files.

They are distributed under the Apache License, Version 2.0, a copy of which is in `LICENSE-APACHE-2.0.txt` beside this file.

- `banking77-lora-v1` (`lora-full/`) and `banking77-gold-1000-v1` (`gold-same-rows/`) were trained on BANKING77's training split (Casanueva et al., 2020; CC BY 4.0; see `examples/data/banking77/DATA_CARD.md`).
- `banking77-distilled-v1` (`distill-student/`) was trained on BANKING77 training messages labelled by **Qwen3-4B-Instruct-2507**, by Qwen, base model `Qwen/Qwen3-4B-Instruct-2507`, licence at commit `cdbee75f17c01a7cc42f958dc650907174af0554` (Apache License, Version 2.0; `unsloth/Qwen3-4B-Instruct-2507-GGUF` at revision `a06e946bb6b655725eafa393f4a9745d460374c9`), recorded in `teacher-labels/`.

Each adapter's `manifest.json` names the script that trained it, its settings and the SHA-256 of each file.
