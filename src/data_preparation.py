"""
Data preparation module.

Responsible for:
  1. Loading the raw dataset (from Hugging Face Hub or a local file)
  2. Formatting examples into instruction/response text using a chat template
  3. Splitting into train/val
  4. Saving processed splits to disk (data/processed/)

Swap the dataset in config/config.yaml and this module handles the rest —
no need to touch train.py or model_utils.py.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset, Dataset

from src.config_utils import load_config


def load_raw_dataset(cfg: dict):
    """Load dataset from Hugging Face Hub (falls back to local jsonl if configured)."""
    dataset_name = cfg["data"]["dataset_name"]
    raw_path = Path(cfg["data"]["raw_path"])

    if raw_path.exists():
        print(f"Loading local raw dataset from {raw_path}")
        return load_dataset("json", data_files=str(raw_path), split="train")

    print(f"Downloading dataset '{dataset_name}' from Hugging Face Hub")
    return load_dataset(dataset_name, split="train")


def format_example(example: dict, cfg: dict) -> dict:
    """Convert a raw (instruction, response) pair into a single training text field
    using a simple, model-agnostic instruction template.

    For chat/instruct models you may prefer Unsloth's `get_chat_template` instead —
    this template is intentionally simple so the pipeline works with any base model.
    """
    instruction_field = cfg["data"]["instruction_field"]
    response_field = cfg["data"]["response_field"]

    instruction = example.get(instruction_field, "").strip()
    response = example.get(response_field, "").strip()

    text = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}"
    )
    return {"text": text}


def prepare_and_save(cfg: dict) -> None:
    raw_dataset = load_raw_dataset(cfg)

    formatted = raw_dataset.map(
        lambda ex: format_example(ex, cfg),
        remove_columns=raw_dataset.column_names,
    )

    split = formatted.train_test_split(
        test_size=cfg["data"]["val_split_ratio"],
        seed=cfg["data"]["seed"],
    )
    train_data, val_data = split["train"], split["test"]

    train_path = Path(cfg["data"]["processed_train_path"])
    val_path = Path(cfg["data"]["processed_val_path"])
    train_path.parent.mkdir(parents=True, exist_ok=True)

    _save_jsonl(train_data, train_path)
    _save_jsonl(val_data, val_path)

    print(f"Saved {len(train_data)} train examples -> {train_path}")
    print(f"Saved {len(val_data)} val examples -> {val_path}")


def _save_jsonl(dataset: Dataset, path: Path) -> None:
    with open(path, "w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare_and_save(cfg)


if __name__ == "__main__":
    main()
