"""
run_contamination_summary.py
=============================
Summarise per-model contamination test results.

Folder structure expected:
    results_folder/ablation/contamination_tests/
        <model>/
            results.csv

Each CSV columns:
    vignette_id, category, mask_type, contamination_level,
    rouge_l_f1, rouge_l_precision, rouge_l_recall,
    ngram_overlap_8, ngram_overlap_13,
    longest_common_substring_ratio,
    exact_match, reference_word_count, generated_word_count,
    generated_text_preview, reference_text_preview

Output:
    results_folder/ablation/_results/contamination_tests_summary.xlsx
        Sheet 1: Per-model summary
        Sheet 2: By mask type
        Sheet 3: By category
"""

from pathlib import Path

import pandas as pd
import json

from icd11_utils import reindex_by_model_order, save_excel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(file):
    with open(file) as f:
        config_dict = json.load(f)
    return config_dict

config_dict = load_config(file=Path(__file__).parents[1].joinpath("config_paths.json"))["hpc"]
BASE_PATH = Path(config_dict['base_path'])
RESULTS_FOLDER = BASE_PATH / "results_resubmission" / "ablation"
CONTAMINATION_DIR = RESULTS_FOLDER / "contamination_tests"
OUTPUT_DIR = RESULTS_FOLDER / "_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def summarize_group(df: pd.DataFrame) -> dict:
    """Compute summary contamination metrics for a group."""
    n = len(df)
    if n == 0:
        return {}
    return {
        "n_tests": n,
        "rouge_l_f1_mean": round(df["rouge_l_f1"].mean(), 4),
        "rouge_l_f1_sd": round(df["rouge_l_f1"].std(), 4),
        "rouge_l_f1_max": round(df["rouge_l_f1"].max(), 4),
        "ngram_overlap_13_mean": round(df["ngram_overlap_13"].mean(), 4),
        "ngram_overlap_13_max": round(df["ngram_overlap_13"].max(), 4),
        "lcs_ratio_mean": round(df["longest_common_substring_ratio"].mean(), 4),
        "lcs_ratio_max": round(df["longest_common_substring_ratio"].max(), 4),
        "n_HIGH": int((df["contamination_level"] == "HIGH").sum()),
        "n_MODERATE": int((df["contamination_level"] == "MODERATE").sum()),
        "n_LOW": int((df["contamination_level"] == "LOW").sum()),
    }


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
model_dirs = sorted([
    d for d in CONTAMINATION_DIR.iterdir()
    if d.is_dir() and (d / "results.csv").exists()
])

if not model_dirs:
    raise FileNotFoundError(
        f"No model folders with results.csv found under {CONTAMINATION_DIR}."
    )

print(f"Found {len(model_dirs)} model(s): {[d.name for d in model_dirs]}\n")

all_data = {}
for model_dir in model_dirs:
    llm = model_dir.name
    df = pd.read_csv(model_dir / "results.csv")
    all_data[llm] = df
    print(f"  {llm}: {len(df)} test cases")

# ---------------------------------------------------------------------------
# Build sheets
# ---------------------------------------------------------------------------
model_records = []
for llm, df in all_data.items():
    record = {"model": llm}
    record.update(summarize_group(df))
    model_records.append(record)
df_models = pd.DataFrame(model_records).set_index("model")
df_models = reindex_by_model_order(df_models)
df_models["rouge_l_f1_formatted"] = df_models.apply(
    lambda r: f"{r['rouge_l_f1_mean']:.4f} ± {r['rouge_l_f1_sd']:.4f}", axis=1
)

mask_records = []
for llm, df in all_data.items():
    for mask_type, group in df.groupby("mask_type"):
        record = {"model": llm, "mask_type": mask_type}
        record.update(summarize_group(group))
        mask_records.append(record)
df_mask = pd.DataFrame(mask_records).set_index(["model", "mask_type"])

cat_records = []
for llm, df in all_data.items():
    for cat, group in df.groupby("category"):
        record = {"model": llm, "category": cat}
        record.update(summarize_group(group))
        cat_records.append(record)
df_cat = pd.DataFrame(cat_records).set_index(["model", "category"])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_excel(
    {"Per Model": df_models, "By Mask Type": df_mask, "By Category": df_cat},
    OUTPUT_DIR / "contamination_tests_summary.xlsx",
)

for label, df_sheet in [("PER-MODEL SUMMARY", df_models), ("BY MASK TYPE", df_mask), ("BY CATEGORY", df_cat)]:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(df_sheet.to_string())

# ---------------------------------------------------------------------------
# Manuscript interpretation
# ---------------------------------------------------------------------------
total_high = df_models["n_HIGH"].sum()
total_moderate = df_models["n_MODERATE"].sum()
total_tests = df_models["n_tests"].sum()

print(f"\n{'='*60}\nINTERPRETATION (for manuscript)\n{'='*60}")
if total_high == 0 and total_moderate == 0:
    print(
        f"Across all {len(all_data)} models and {int(total_tests)} test cases, "
        "no model exceeded the HIGH or MODERATE contamination thresholds "
        "(13-gram overlap > 0.10 or ROUGE-L F1 > 0.70 for HIGH; "
        "8-gram overlap > 0.10 or ROUGE-L F1 > 0.40 for MODERATE). "
        "This suggests the models did not memorize the clinical vignettes."
    )
elif total_high > 0:
    print(f"WARNING: {int(total_high)} test cases across models showed HIGH contamination.")
    print("Investigate per-model and per-vignette results for details.")
else:
    print(f"{int(total_moderate)} test cases showed MODERATE contamination signal.")
    print("Some familiarity detected but below strong memorization thresholds.")
