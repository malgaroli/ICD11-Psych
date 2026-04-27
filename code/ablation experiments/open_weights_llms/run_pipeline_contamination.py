"""
Contamination Completion Test for Clinical Vignettes
=====================================================

Tests whether LLMs have memorized clinical vignettes by prompting them
to complete masked portions and measuring overlap with the original text.

Uses the existing LLMModel class for inference (same as PREVIA pipeline).

Based on: Golchin & Surdeanu (2024), "Time Travel in LLMs: Tracing Data
Contamination in Large Language Models" (arXiv:2308.08493)

Usage (HPC):
    python contamination_test.py \
        --config config_paths.json \
        --deployment hpc \
        --model llama33_70B \
        --output results/contamination_llama33.json \
        --batch-size 4

Requirements:
    pip install pandas rouge-score transformers accelerate bitsandbytes
    + LLMModel.py in the same directory or on PYTHONPATH
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from LLMModel import LLMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_config(file: Path) -> dict:
    with open(file, "r") as f:
        return json.load(f)


def load_api_key(path: Path = Path("huggingface_key.txt")) -> str:
    """Load HuggingFace API key from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def resolve_model_path(config_path: Path, deployment: str, model_key: str) -> str:
    """
    Resolve a model key (e.g. 'llama33_70B') to the full model path
    from config_paths.json.

    Looks up '{model_key}_path' in the deployment config.
    If the key isn't found, treats model_key as a direct path/HF name.
    """
    config_dict = load_config(file=config_path)[deployment]
    config_key = f"{model_key}_path"

    if config_key in config_dict:
        model_path = config_dict[config_key]
        log.info(f"Resolved model key '{model_key}' -> {model_path}")
        return model_path
    elif model_key in config_dict:
        # Allow passing the full key name too (e.g. 'llama33_70B_path')
        model_path = config_dict[model_key]
        log.info(f"Resolved model key '{model_key}' -> {model_path}")
        return model_path
    else:
        # Treat as direct path or HuggingFace model name
        available = [k.replace("_path", "") for k in config_dict if k.endswith("_path") and k != "base_path"]
        log.warning(
            f"Model key '{model_key}' not found in config. "
            f"Available keys: {available}. "
            f"Treating '{model_key}' as a direct model path."
        )
        return model_key


def load_vignettes(config_path: Path, deployment: str, categories: list[str]) -> list[dict]:
    """Load vignettes from CSV using the same logic as PromptBuilder."""
    config_dict = load_config(file=config_path)[deployment]
    base_path = Path(config_dict["base_path"])
    data_path = base_path / "data" / "Data_final_updated.csv"

    log.info(f"Loading vignettes from {data_path}")
    data = pd.read_csv(data_path, encoding="utf-8")

    vignettes = []
    for category in categories:
        df_cat = data[data["Category"] == category]
        for _, row in df_cat.iterrows():
            vignette_id = row["Vignette ID"]
            referral = str(row["Referral"])
            presenting_symptoms = str(row["Presenting Symptoms"])
            add_background_info = str(row["Additional Background Information"])

            full_vignette = (
                f"Referral: {referral}\n"
                f"Presenting Symptoms: {presenting_symptoms}\n"
                f"Additional Background Information: {add_background_info}"
            )

            vignettes.append({
                "id": f"{category}_{vignette_id}",
                "category": category,
                "referral": referral,
                "presenting_symptoms": presenting_symptoms,
                "add_background_info": add_background_info,
                "full_vignette": full_vignette,
            })

    log.info(f"Loaded {len(vignettes)} vignettes across {categories}")
    return vignettes


# ═══════════════════════════════════════════════════════════════════
# 2. MASKING STRATEGIES
# ═══════════════════════════════════════════════════════════════════

def mask_by_ratio(text: str, mask_ratio: float = 0.5) -> tuple[str, str]:
    """Split text into visible prefix and masked suffix."""
    words = text.split()
    split_point = int(len(words) * (1 - mask_ratio))
    split_point = max(1, split_point)
    prefix = " ".join(words[:split_point])
    suffix = " ".join(words[split_point:])
    return prefix, suffix


def build_test_cases(vignette: dict, mask_ratio: float = 0.5) -> list[dict]:
    """
    Create three masking conditions per vignette:
      1. mask_symptoms: given referral, complete presenting symptoms
      2. mask_background: given referral + symptoms, complete background
      3. mask_second_half: given first half, complete rest
    """
    tests = []

    tests.append({
        "mask_type": "mask_symptoms",
        "prompt_context": f"Referral: {vignette['referral']}",
        "expected_completion": vignette["presenting_symptoms"],
        "instruction": (
            "The following is the beginning of a clinical case vignette. "
            "Please complete the rest of the vignette, continuing with the "
            "Presenting Symptoms section. Reproduce the original text as "
            "closely as possible.\n\n"
        ),
    })

    tests.append({
        "mask_type": "mask_background",
        "prompt_context": (
            f"Referral: {vignette['referral']}\n"
            f"Presenting Symptoms: {vignette['presenting_symptoms']}"
        ),
        "expected_completion": vignette["add_background_info"],
        "instruction": (
            "The following is the beginning of a clinical case vignette. "
            "Please complete the rest of the vignette, continuing with the "
            "Additional Background Information section. Reproduce the "
            "original text as closely as possible.\n\n"
        ),
    })

    prefix, suffix = mask_by_ratio(vignette["full_vignette"], mask_ratio=mask_ratio)
    tests.append({
        "mask_type": "mask_second_half",
        "prompt_context": prefix,
        "expected_completion": suffix,
        "instruction": (
            "The following is the beginning of a clinical case vignette. "
            "Please complete the rest of the vignette exactly as you "
            "believe it continues. Reproduce the original text as closely "
            "as possible.\n\n"
        ),
    })

    return tests


# ═══════════════════════════════════════════════════════════════════
# 3. PROMPT FORMATTING (compatible with LLMModel)
# ═══════════════════════════════════════════════════════════════════

def build_prompts_for_llmmodel(flat_tests: list[dict]) -> list[dict]:
    """
    Build prompts in the format expected by LLMModel.process_all_batches():
      [{"id": ..., "prompt": [{"role": "user", "content": ...}]}, ...]
    """
    prompts = []
    for test in flat_tests:
        user_content = f"{test['instruction']}{test['prompt_context']}\n\nContinuation:"
        prompts.append({
            "id": f"{test['vignette_id']}__{test['mask_type']}",
            "prompt": [{"role": "user", "content": user_content}],
        })
    return prompts


# ═══════════════════════════════════════════════════════════════════
# 4. OVERLAP METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_overlap_metrics(generated: str, reference: str) -> dict:
    """
    Compute overlap metrics between generated completion and original.

    Returns:
      - rouge_l_f1, rouge_l_precision, rouge_l_recall
      - ngram_overlap_8, ngram_overlap_13
      - exact_match
      - longest_common_substring_ratio
      - reference_word_count, generated_word_count
    """
    from rouge_score import rouge_scorer

    gen_norm = " ".join(generated.lower().split())
    ref_norm = " ".join(reference.lower().split())

    metrics = {}

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(ref_norm, gen_norm)
    metrics["rouge_l_f1"] = round(scores["rougeL"].fmeasure, 4)
    metrics["rouge_l_precision"] = round(scores["rougeL"].precision, 4)
    metrics["rouge_l_recall"] = round(scores["rougeL"].recall, 4)

    # N-gram overlap (8-gram and 13-gram)
    ref_words = ref_norm.split()
    gen_words = gen_norm.split()

    for n in [8, 13]:
        if len(ref_words) < n:
            metrics[f"ngram_overlap_{n}"] = 0.0
            continue
        ref_ngrams = set(
            tuple(ref_words[i : i + n]) for i in range(len(ref_words) - n + 1)
        )
        gen_ngrams = set(
            tuple(gen_words[i : i + n]) for i in range(len(gen_words) - n + 1)
        )
        if len(ref_ngrams) == 0:
            metrics[f"ngram_overlap_{n}"] = 0.0
        else:
            overlap = len(ref_ngrams & gen_ngrams)
            metrics[f"ngram_overlap_{n}"] = round(overlap / len(ref_ngrams), 4)

    # Exact match
    metrics["exact_match"] = int(gen_norm == ref_norm)

    # Longest common substring ratio (word-level)
    metrics["longest_common_substring_ratio"] = round(
        _lcs_ratio(gen_norm, ref_norm), 4
    )

    # Word counts
    metrics["reference_word_count"] = len(ref_words)
    metrics["generated_word_count"] = len(gen_words)

    return metrics


def _lcs_ratio(s1: str, s2: str) -> float:
    """Longest common substring (word-level) as ratio of shorter string."""
    words1 = s1.split()
    words2 = s2.split()
    if not words1 or not words2:
        return 0.0
    m, n = len(words1), len(words2)
    max_len = 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                curr[j] = prev[j - 1] + 1
                max_len = max(max_len, curr[j])
        prev = curr
    return max_len / min(m, n) if min(m, n) > 0 else 0.0


def interpret_contamination(metrics: dict) -> str:
    """
    Heuristic contamination classification.
    Thresholds from Brown et al. (2020) and Golchin & Surdeanu (2024).
    """
    if metrics["ngram_overlap_13"] > 0.1 or metrics["rouge_l_f1"] > 0.7:
        return "HIGH"
    elif metrics["ngram_overlap_8"] > 0.1 or metrics["rouge_l_f1"] > 0.4:
        return "MODERATE"
    else:
        return "LOW"


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_contamination_test(
    config_path: Path,
    deployment: str,
    model_path: str,
    categories: list[str],
    output_path: Path,
    mask_ratio: float = 0.5,
    max_new_tokens: int = 512,
    batch_size: int = 4,
):
    """Full contamination test pipeline using LLMModel."""

    # Load HuggingFace API key
    config_dict = load_config(file=config_path)[deployment]
    base_path = Path(config_dict["base_path"])
    api_key_huggingface = load_api_key(path=base_path.parent.joinpath("huggingface_key.txt"))

    # Load vignettes
    vignettes = load_vignettes(config_path, deployment, categories)

    # Build test cases
    flat_tests = []
    for vig in vignettes:
        tests = build_test_cases(vig, mask_ratio=mask_ratio)
        for test in tests:
            test["vignette_id"] = vig["id"]
            test["category"] = vig["category"]
        flat_tests.extend(tests)

    log.info(f"Created {len(flat_tests)} test cases from {len(vignettes)} vignettes")

    # Build prompts in LLMModel format
    all_prompts = build_prompts_for_llmmodel(flat_tests)

    # Run inference via LLMModel
    log.info(f"Loading model from {model_path}...")
    model = LLMModel(
        model_path,
        max_new_tokens=max_new_tokens,
        api_key_huggingface=api_key_huggingface,
    )

    log.info(f"Running inference on {len(all_prompts)} prompts (batch_size={batch_size})...")
    raw_responses = model.process_all_batches(all_prompts, batch_size=batch_size)

    # raw_responses is list of dicts: [{id: response}, ...]
    response_lookup = {}
    for resp_dict in raw_responses:
        response_lookup.update(resp_dict)

    # Compute metrics
    results = []
    for test in flat_tests:
        test_id = f"{test['vignette_id']}__{test['mask_type']}"
        completion = response_lookup.get(test_id, "")
        if completion is None:
            log.warning(f"No response for {test_id}, skipping.")
            completion = ""

        metrics = compute_overlap_metrics(completion, test["expected_completion"])
        contamination_level = interpret_contamination(metrics)

        results.append({
            "vignette_id": test["vignette_id"],
            "category": test["category"],
            "mask_type": test["mask_type"],
            "contamination_level": contamination_level,
            **metrics,
            "generated_text_preview": completion[:300],
            "reference_text_preview": test["expected_completion"][:300],
        })

    # Aggregate
    df_results = pd.DataFrame(results)

    summary = {
        "overall": _summarize_group(df_results),
        "by_category": {
            cat: _summarize_group(df_results[df_results["category"] == cat])
            for cat in categories
        },
        "by_mask_type": {
            mt: _summarize_group(df_results[df_results["mask_type"] == mt])
            for mt in df_results["mask_type"].unique()
        },
    }

    # Save
    output = {
        "metadata": {
            "model": model_path,
            "categories": categories,
            "mask_ratio": mask_ratio,
            "n_vignettes": len(vignettes),
            "n_test_cases": len(flat_tests),
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": summary,
        "per_vignette": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = output_path.with_suffix(".csv")
    df_results.to_csv(csv_path, index=False)

    log.info(f"Results saved to {output_path} and {csv_path}")
    _print_summary(summary, model_path)

    return output


def _summarize_group(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "n": len(df),
        "rouge_l_f1_mean": round(df["rouge_l_f1"].mean(), 4),
        "rouge_l_f1_std": round(df["rouge_l_f1"].std(), 4),
        "rouge_l_f1_max": round(df["rouge_l_f1"].max(), 4),
        "ngram_overlap_8_mean": round(df["ngram_overlap_8"].mean(), 4),
        "ngram_overlap_13_mean": round(df["ngram_overlap_13"].mean(), 4),
        "lcs_ratio_mean": round(df["longest_common_substring_ratio"].mean(), 4),
        "contamination_high_n": int((df["contamination_level"] == "HIGH").sum()),
        "contamination_moderate_n": int((df["contamination_level"] == "MODERATE").sum()),
        "contamination_low_n": int((df["contamination_level"] == "LOW").sum()),
    }


def _print_summary(summary: dict, model_name: str):
    print("\n" + "=" * 70)
    print(f"  CONTAMINATION TEST RESULTS: {model_name}")
    print("=" * 70)

    ov = summary["overall"]
    print(f"\n  Overall ({ov['n']} test cases):")
    print(f"    ROUGE-L F1:      {ov['rouge_l_f1_mean']:.4f} +/- {ov['rouge_l_f1_std']:.4f} (max: {ov['rouge_l_f1_max']:.4f})")
    print(f"    8-gram overlap:  {ov['ngram_overlap_8_mean']:.4f}")
    print(f"    13-gram overlap: {ov['ngram_overlap_13_mean']:.4f}")
    print(f"    LCS ratio:       {ov['lcs_ratio_mean']:.4f}")
    print(f"    Contamination:   HIGH={ov['contamination_high_n']}, "
          f"MODERATE={ov['contamination_moderate_n']}, LOW={ov['contamination_low_n']}")

    print(f"\n  By Category:")
    for cat, stats in summary["by_category"].items():
        if not stats:
            continue
        print(f"    {cat} (n={stats['n']}): ROUGE-L={stats['rouge_l_f1_mean']:.4f}, "
              f"13-gram={stats['ngram_overlap_13_mean']:.4f}, "
              f"HIGH={stats['contamination_high_n']}")

    print(f"\n  By Mask Type:")
    for mt, stats in summary["by_mask_type"].items():
        if not stats:
            continue
        print(f"    {mt} (n={stats['n']}): ROUGE-L={stats['rouge_l_f1_mean']:.4f}, "
              f"13-gram={stats['ngram_overlap_13_mean']:.4f}, "
              f"HIGH={stats['contamination_high_n']}")

    print("\n  Interpretation:")
    if ov["contamination_high_n"] == 0 and ov["contamination_moderate_n"] == 0:
        print("    No evidence of memorization detected.")
        print("    The model does not appear to have seen these vignettes during training.")
    elif ov["contamination_high_n"] > 0:
        print(f"    WARNING: {ov['contamination_high_n']} test cases show HIGH contamination signal.")
        print("    The model may have memorized some or all of these vignettes.")
    else:
        print(f"    {ov['contamination_moderate_n']} test cases show MODERATE signal.")
        print("    Some familiarity detected but below strong memorization threshold.")

    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# 6. CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Contamination completion test for clinical vignettes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal — output auto-created at contamination_analysis/llama33_70B/
  python contamination_test.py \\
      --config config_paths.json \\
      --model llama33_70B

  python contamination_test.py \\
      --config config_paths.json \\
      --model mistral_large \\
      --batch-size 2

  # Custom output path (overrides auto-generated)
  python contamination_test.py \\
      --config config_paths.json \\
      --model llama31_8B \\
      --output my_results/custom_output.json

Available model keys (from your config):
  llama31_8B, llama33_70B, llama32_3B, mistral_7B, mistral_large,
  gemma3, deepseek_70B, Qwen25_72B, Qwen25_32B, gpt_20B, gpt_120B, ...

References:
  - Golchin & Surdeanu (2024). Time Travel in LLMs. arXiv:2308.08493
  - Golchin & Surdeanu (2024). Data Contamination Quiz. TACL.
  - Brown et al. (2020). GPT-3: 13-gram contamination threshold.
  - Shi et al. (2024). Detecting Pretraining Data from LLMs. ICLR.
        """,
    )

    parser.add_argument("--config", type=Path, required=True,
                        help="Path to config_paths.json")
    parser.add_argument("--deployment", type=str, default="hpc",
                        help="Deployment type key in config (default: hpc)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model key from config (e.g. 'llama33_70B') or direct HF path")
    parser.add_argument("--categories", nargs="+", default=["Mood", "Anxiety", "Stress"],
                        help="Vignette categories (default: Mood Anxiety Stress)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: contamination_analysis/<model>/results.json)")
    parser.add_argument("--mask-ratio", type=float, default=0.5,
                        help="Fraction of text to mask (default: 0.5)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max new tokens for completion (default: 512)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference (default: 4)")

    args = parser.parse_args()

    # Resolve model key to path
    model_path = resolve_model_path(args.config, args.deployment, args.model)

    # Auto-generate output path if not specified
    if args.output is None:
        # Derive a clean folder name from the model key
        model_folder = args.model.replace("/", "_").replace("\\", "_")
        output_dir = Path("contamination_analysis") / model_folder
        output_path = output_dir / "results.json"
    else:
        output_path = args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Output will be saved to {output_path}")

    run_contamination_test(
        config_path=args.config,
        deployment=args.deployment,
        model_path=model_path,
        categories=args.categories,
        output_path=output_path,
        mask_ratio=args.mask_ratio,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()