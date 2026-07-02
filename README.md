# Domain-Specific LLM Fine-Tuning (QLoRA)

Fine-tunes an open-source LLM (Llama 3.1/3.2, default) on a domain-specific instruction
dataset using QLoRA (4-bit quantization + LoRA adapters) via Unsloth, then serves it
through a Streamlit chat interface.

This repo is intentionally structured like a production ML project, not a single
notebook — config-driven, modular, testable, and deployable.

## Project Structure

```
llm-finetune-project/
├── config/
│   └── config.yaml          # all hyperparameters & paths in one place
├── data/
│   ├── raw/                 # original dataset (jsonl/csv)
│   └── processed/           # cleaned, formatted train/val splits
├── src/
│   ├── data_preparation.py  # load, clean, format, split dataset
│   ├── model_utils.py       # model/tokenizer loading (Unsloth + LoRA config)
│   ├── train.py             # training entrypoint (SFTTrainer)
│   ├── inference.py         # load fine-tuned adapter, run generation
│   └── evaluate.py          # quantitative eval (ROUGE/exact-match) + qualitative samples
├── app/
│   └── streamlit_app.py     # chat UI over the fine-tuned model
├── scripts/
│   ├── run_training.sh      # one-command training entrypoint
│   └── run_app.sh
├── models/                  # saved LoRA adapters (gitignored, populated after training)
├── logs/                    # training logs / metrics (gitignored)
├── tests/
│   └── test_data_preparation.py
├── requirements.txt
└── README.md
```

## Why modular > single notebook

- **`config.yaml`** — change model, dataset, or hyperparameters without touching code.
- **`data_preparation.py`** — dataset logic is reusable across any future fine-tuning project (swap the dataset, keep the pipeline).
- **`model_utils.py`** — model loading/LoRA setup is isolated, so switching base models (Llama → Mistral → Qwen) is a one-line change.
- **`train.py` / `inference.py` / `evaluate.py`** — clean separation between training, serving, and evaluation, matching how real ML teams organize repos.
- **`app/`** — a deployable interface, not just a Colab cell.

## Quickstart

```bash
# 1. Environment
pip install -r requirements.txt --break-system-packages

# 2. Prepare data (edit config/config.yaml first: dataset name, format)
python -m src.data_preparation

# 3. Train (QLoRA fine-tuning)
python -m src.train --config config/config.yaml

# 4. Evaluate
python -m src.evaluate --config config/config.yaml

# 5. Run the demo app
streamlit run app/streamlit_app.py
```

Or just: `bash scripts/run_training.sh`

## Hardware

Runs on a free Google Colab T4 GPU (16GB) for models up to 8B parameters in 4-bit.
If running locally without a GPU, set `use_colab: true` behavior isn't needed — just
run the notebook version referenced in `scripts/colab_notebook_link.txt`.

## Results (fill in after training)

| Metric | Base Model | Fine-Tuned |
|---|---|---|
| Exact match | - | - |
| ROUGE-L | - | - |
| Domain accuracy (manual eval, n=50) | - | - |

## Resume bullet (fill in after training)

> Fine-tuned Llama 3.1 8B using QLoRA (4-bit NF4, r=16) on a [N]-example domain-specific
> instruction dataset via Unsloth; reduced training cost ~4x vs. full fine-tuning and
> deployed via Streamlit with adapter published to Hugging Face Hub.
