"""
Contamination Completion Test for Clinical Vignettes (API Version)
==================================================================

Tests whether LLMs have memorized clinical vignettes by prompting them
to complete masked portions and measuring overlap with the original text.

Uses API calls (GPT, Claude, Gemini) via NYU Langone Kong proxy.

Based on: Golchin & Surdeanu (2024), "Time Travel in LLMs: Tracing Data
Contamination in Large Language Models" (arXiv:2308.08493)

Usage:
    python contamination_test_api.py

Requirements:
    pip install pandas rouge-score openai
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEPLOYMENT_TYPE = "hpc"
CATEGORIES = ["Mood", "Anxiety", "Stress"]
MODELS = ["gemini_25"]


# ═══════════════════════════════════════════════════════════════════
# 1. CONFIG & API
# ═══════════════════════════════════════════════════════════════════

def load_config(file: Path) -> dict:
    with open(file, "r") as f:
        return json.load(f)


def load_api_key(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def run_API_LLM(messages, API_KEY, model_version):
    """Route to the correct API based on model version string."""
    if "gpt" in model_version:
        return _run_api_call(messages, API_KEY, model_version)
    elif ("sonnet" in model_version) or ("opus" in model_version):
        return _run_api_call(messages, API_KEY, model_version)
    elif "gemini" in model_version:
        return _run_api_call(messages, API_KEY, model_version)
    else:
        raise ValueError(f"Unknown model version: {model_version}")


def _run_api_call(messages, API_KEY, model_version):
    """Generic API call via NYU Langone Kong proxy (OpenAI-compatible)."""
    from openai import OpenAI

    endpoint = f"https://kong-api.prod1.nyumc.org/{model_version}"
    deployment = "required-but-not-used-by-openai-lib"

    client = OpenAI(
        base_url=endpoint,
        api_key=API_KEY,
        default_headers={"api-key": API_KEY},
    )

    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
    )

    assert completion.choices, "Expected at least one choice in the response."
    return completion.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_vignettes(base_path: Path, categories: list[str]) -> list[dict]:
    """Load vignettes from CSV."""
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
# 3. MASKING STRATEGIES
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
# 5. SUMMARY & PRINTING
# ═══════════════════════════════════════════════════════════════════

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
# 6. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_contamination_test_for_model(
    llm_key: str,
    model_version: str,
    base_path: Path,
    API_KEY: str,
    vignettes: list[dict],
    mask_ratio: float = 0.5,
):
    """Run contamination test for a single model via API."""

    # Build test cases
    flat_tests = []
    for vig in vignettes:
        tests = build_test_cases(vig, mask_ratio=mask_ratio)
        for test in tests:
            test["vignette_id"] = vig["id"]
            test["category"] = vig["category"]
        flat_tests.extend(tests)

    log.info(f"Created {len(flat_tests)} test cases from {len(vignettes)} vignettes")

    # Set up output and backup directories
    output_dir = base_path / "contamination_analysis" / llm_key
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir / "backups"
    backup_dir.mkdir(exist_ok=True)

    # Run inference via API, one prompt at a time
    log.info(f"Running inference for {llm_key} ({model_version})...")
    results = []

    for i, test in enumerate(flat_tests):
        test_id = f"{test['vignette_id']}__{test['mask_type']}"
        user_content = f"{test['instruction']}{test['prompt_context']}\n\nContinuation:"
        messages = [{"role": "user", "content": user_content}]

        log.info(f"  [{i+1}/{len(flat_tests)}] {test_id}")

        try:
            completion = run_API_LLM(messages, API_KEY, model_version)
        except Exception as e:
            log.error(f"  API call failed for {test_id}: {e}")
            completion = ""

        # Save backup
        backup_data = {
            "test_id": test_id,
            "prompt": user_content,
            "response": completion,
        }
        pd.DataFrame([backup_data]).to_csv(
            backup_dir / f"contamination_{llm_key}_{test_id}.csv",
            index=False,
        )

        # Compute metrics
        if completion:
            metrics = compute_overlap_metrics(completion, test["expected_completion"])
        else:
            metrics = compute_overlap_metrics("", test["expected_completion"])

        contamination_level = interpret_contamination(metrics)

        results.append({
            "vignette_id": test["vignette_id"],
            "category": test["category"],
            "mask_type": test["mask_type"],
            "contamination_level": contamination_level,
            **metrics,
            "generated_text_preview": (completion or "")[:300],
            "reference_text_preview": test["expected_completion"][:300],
        })

    # Aggregate
    df_results = pd.DataFrame(results)

    summary = {
        "overall": _summarize_group(df_results),
        "by_category": {
            cat: _summarize_group(df_results[df_results["category"] == cat])
            for cat in CATEGORIES
        },
        "by_mask_type": {
            mt: _summarize_group(df_results[df_results["mask_type"] == mt])
            for mt in df_results["mask_type"].unique()
        },
    }

    # Save JSON
    output = {
        "metadata": {
            "model_key": llm_key,
            "model_version": model_version,
            "categories": CATEGORIES,
            "mask_ratio": mask_ratio,
            "n_vignettes": len(vignettes),
            "n_test_cases": len(flat_tests),
            "timestamp": datetime.now().isoformat(),
        },
        "summary": summary,
        "per_vignette": results,
    }

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save CSV
    csv_path = output_dir / "results.csv"
    df_results.to_csv(csv_path, index=False)

    log.info(f"Results saved to {json_path} and {csv_path}")
    _print_summary(summary, f"{llm_key} ({model_version})")

    return output


def main():
    # Load config
    config_path = Path(__file__).parent / "config_paths.json"
    config_dict = load_config(file=config_path)[DEPLOYMENT_TYPE]
    base_path = Path(config_dict["base_path"])

    # Load API key
    API_KEY = load_api_key(path=base_path / "token.txt")

    # Load vignettes once (shared across models)
    vignettes = load_vignettes(base_path, CATEGORIES)

    # Loop over models
    for llm_key in MODELS:
        model_version_key = f"{llm_key}_path"
        if model_version_key not in config_dict:
            log.error(f"Model key '{llm_key}' not found in config. "
                      f"Expected key '{model_version_key}'. Skipping.")
            continue

        model_version = config_dict[model_version_key]
        log.info(f"\n{'='*70}")
        log.info(f"Starting contamination test for: {llm_key} -> {model_version}")
        log.info(f"{'='*70}")

        try:
            run_contamination_test_for_model(
                llm_key=llm_key,
                model_version=model_version,
                base_path=base_path,
                API_KEY=API_KEY,
                vignettes=vignettes,
                mask_ratio=0.5,
            )
        except Exception as e:
            log.error(f"Failed for {llm_key}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()