"""
Inference module.

Loads the fine-tuned adapter and exposes a single `generate()` function used
by both evaluate.py and the Streamlit app — one code path, no duplicated
prompt-formatting logic.
"""

import argparse

from src.config_utils import load_config
from src.model_utils import load_model_for_inference


def build_prompt(instruction: str) -> str:
    """Must match the template used in data_preparation.py's format_example()."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate(model, tokenizer, instruction: str, cfg: dict) -> str:
    prompt = build_prompt(instruction)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    inf_cfg = cfg["inference"]
    outputs = model.generate(
        **inputs,
        max_new_tokens=inf_cfg["max_new_tokens"],
        temperature=inf_cfg["temperature"],
        top_p=inf_cfg["top_p"],
        use_cache=True,
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Strip the prompt back off so only the generated response is returned
    return decoded.split("### Response:\n", 1)[-1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--instruction", required=True, help="Prompt to send to the model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter_path = cfg["training"]["final_adapter_dir"]

    model, tokenizer = load_model_for_inference(cfg, adapter_path)
    response = generate(model, tokenizer, args.instruction, cfg)

    print(f"\nInstruction: {args.instruction}\nResponse: {response}")


if __name__ == "__main__":
    main()
