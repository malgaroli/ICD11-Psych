"""
run_essential_features.py
==========================
Classification metrics for the essential features ablation experiment
(English only).

Same metrics as run_classification_metrics.py but pointed at:
    results_folder/ablation/essential_features/<model>/english/*results.csv

Output:
    results_folder/ablation/_results/essential_features_metrics.xlsx
        Sheets: Overall, Anxiety, Mood, Stress, PerClass_<model>
"""

from pathlib import Path
import json
from icd11_utils import (
    apply_label_corrections,
    build_metrics_sheets,
    build_per_class_sheets,
    load_results,
    print_sheets,
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
EF_DIR = RESULTS_FOLDER / "essential_features"
OUTPUT_DIR = RESULTS_FOLDER / "_results"
OUTPUT_DIR.mkdir(exist_ok=True)

LANGUAGE = "english"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
files = sorted(EF_DIR.glob(f"*/{LANGUAGE}/*results.csv"))
files = [f for f in files if "results_old" not in str(f)]

if not files:
    raise FileNotFoundError(
        f"No *results.csv files found under {EF_DIR}/*/english/. "
        "Check your EF_DIR path."
    )

print(f"Found {len(files)} result files:")
for f in files:
    print(f"  {f}")

combined = load_results(files, llm_part_index=-3)
combined = apply_label_corrections(combined)

print(f"\nTotal vignettes: {len(combined)}")
print(f"Models:          {combined['llm'].unique().tolist()}")
print(f"Categories:      {combined['Category'].unique().tolist()}")

sheets = build_metrics_sheets(combined)
per_class_sheets = build_per_class_sheets(combined)

all_sheets = {**sheets, **per_class_sheets}
save_excel(all_sheets, OUTPUT_DIR / "essential_features_metrics.xlsx")
print_sheets(sheets)

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
