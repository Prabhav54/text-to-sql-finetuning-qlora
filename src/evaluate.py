"""
Evaluation module.

Runs the held-out validation set through the fine-tuned model and reports:
  - ROUGE-L (text overlap with reference response)
  - Exact-match rate (for short-answer style tasks)
  - A handful of qualitative sample generations for manual review

This is what turns "I fine-tuned a model" into "I fine-tuned a model and can
show it actually improved" — the metric table in the README is populated
from logs/eval_results.json produced here.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from rouge_score import rouge_scorer

from src.config_utils import load_config
from src.model_utils import load_model_for_inference
from src.inference import generate


def compute_rouge(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return {"rougeL_avg": sum(scores) / len(scores) if scores else 0.0}


def compute_exact_match(predictions: list, references: list) -> dict:
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if pred.strip().lower() == ref.strip().lower()
    )
    return {"exact_match": matches / len(predictions) if predictions else 0.0}


def run_evaluation(cfg: dict) -> None:
    adapter_path = cfg["training"]["final_adapter_dir"]
    model, tokenizer = load_model_for_inference(cfg, adapter_path)

    val_dataset = load_dataset(
        "json", data_files=cfg["data"]["processed_val_path"], split="train"
    )

    predictions, references, samples = [], [], []

    for i, row in enumerate(val_dataset):
        text = row["text"]
        instruction, reference = text.split("### Response:\n", 1)
        instruction = instruction.replace("### Instruction:\n", "").strip()
        reference = reference.strip()

        prediction = generate(model, tokenizer, instruction, cfg)

        predictions.append(prediction)
        references.append(reference)

        if i < cfg["evaluation"]["num_qualitative_samples"]:
            samples.append({
                "instruction": instruction,
                "reference": reference,
                "prediction": prediction,
            })

    results = {
        **compute_rouge(predictions, references),
        **compute_exact_match(predictions, references),
        "num_examples": len(predictions),
        "qualitative_samples": samples,
    }

    out_path = Path(cfg["evaluation"]["eval_output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. ROUGE-L: {results['rougeL_avg']:.3f} | "
          f"Exact match: {results['exact_match']:.3f}")
    print(f"Full results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
