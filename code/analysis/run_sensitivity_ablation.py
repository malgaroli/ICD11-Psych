"""
run_sensitivity_ablation.py
============================
Accuracy + Wilson 95% CI for the prompt sensitivity and paraphrase
vignette ablation experiments.

Folder structure expected:
    results_folder/ablation/
        prompt_sensitivity/
            <model>/
                v1/ v2/ v3/
                    *results.csv
        paraphrased_vignettes/
            <model>/
                low/ medium/ high/
                    *results.csv

Output:
    results_folder/ablation/_results/ablation_sensitivity_accuracy_wilson.xlsx
        One sheet: models as rows, 6 condition columns, each showing
        "accuracy [CI_lower, CI_upper]"
"""

from pathlib import Path

import pandas as pd
import json

from icd11_utils import (
    sort_models,
    apply_label_corrections,
    compute_accuracy_wilson,
    load_results,
    save_excel,
)

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
OUTPUT_DIR = RESULTS_FOLDER / "_results"
OUTPUT_DIR.mkdir(exist_ok=True)

ABLATION_TYPES = {
    "prompt_sensitivity": {
        "v1": "prompt_v1",
        "v2": "prompt_v2",
        "v3": "prompt_v3",
    },
    "paraphrased_vignettes": {
        "low": "paraphrase_low",
        "medium": "paraphrase_medium",
        "high": "paraphrase_high",
    },
}

# ---------------------------------------------------------------------------
# Discover models
# ---------------------------------------------------------------------------
all_models = set()
for ablation_folder in ABLATION_TYPES:
    ablation_dir = RESULTS_FOLDER / ablation_folder
    if ablation_dir.is_dir():
        all_models.update(d.name for d in ablation_dir.iterdir() if d.is_dir())

all_models = sorted(all_models)

if not all_models:
    raise FileNotFoundError(
        f"No model folders found under {RESULTS_FOLDER}/prompt_sensitivity/ "
        f"or {RESULTS_FOLDER}/paraphrased_vignettes/."
    )

print(f"Found {len(all_models)} model(s): {all_models}\n")

# ---------------------------------------------------------------------------
# Compute accuracy + Wilson CI per model per condition
# ---------------------------------------------------------------------------
records = []

for llm in all_models:
    row = {"llm": llm}

    for ablation_folder, conditions in ABLATION_TYPES.items():
        for condition_folder, col_name in conditions.items():
            condition_dir = RESULTS_FOLDER / ablation_folder / llm / condition_folder
            files = sorted(condition_dir.glob("*results.csv")) if condition_dir.is_dir() else []
            files = [f for f in files if "results_old" not in str(f)]

            if not files:
                print(f"  WARNING: No results.csv found for {ablation_folder}/{llm}/{condition_folder}")
                row[col_name] = "N/A"
                continue

            combined = load_results(files, llm_part_index=-3)
            combined = apply_label_corrections(combined)
            metrics = compute_accuracy_wilson(combined)
            row[col_name] = metrics["formatted"]

            print(f"  {llm} / {ablation_folder}/{condition_folder}: {metrics['formatted']} (n={len(combined)})")

    records.append(row)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
col_order = [col for conditions in ABLATION_TYPES.values() for col in conditions.values()]
df_results = pd.DataFrame(records).set_index("llm")[col_order]
df_results = df_results.reindex(sort_models(df_results.index.tolist()))

save_excel({"Ablation Results": df_results}, OUTPUT_DIR / "ablation_sensitivity_accuracy_wilson.xlsx")

print(f"\n{'='*60}\nRESULTS\n{'='*60}")
print(df_results.to_string())
print(f"\nSaved to {OUTPUT_DIR}/ablation_sensitivity_accuracy_wilson.xlsx")
