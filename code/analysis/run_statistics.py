"""
run_statistics.py
=================
Orchestration script: run all significance tests for the ICD-11 benchmarking project
and save results to Excel workbooks.

Tests run
---------
1. Wilcoxon signed-rank    — per-vignette LLM correctness (0/1) vs. clinician
                             mean accuracy ([0,1]) — replaces McNemar for
                             LLM-vs-clinician because clinicians are a proportion,
                             not a single binary rater
2. English LLM vs. LLM    — McNemar pairwise (overall + per category)
3. Non-inferiority + TOST  — one-sided NI test AND two one-sided equivalence test,
                             both at δ = 0.10, LLM vs. clinician mean accuracy
4. Top-N vs. random        — proportions z-test per model per category
5. Multilingual McNemar    — pairwise language comparisons per model

Clinician input
---------------
CLINICIAN_FILE = RESULTS_FOLDER / "_results" / "clinicians" / "clinicians_harmonised.csv"

Expected columns:
    Vignette_ID, ID, Category, Language, Ground_Truth_Label, Predicted_Label

Per-vignette clinician accuracy = mean(Predicted_Label == Ground_Truth_Label)
across all clinicians who answered that vignette.

English LLM results
-------------------
RESULTS_FOLDER/english/<model>/english/*results.csv
Must contain: Vignette_ID, Category, Ground_Truth_Label,
              Predicted_Label, Top_1_Accuracy, Top_2_Accuracy, Top_3_Accuracy

Multilingual LLM results
------------------------
RESULTS_FOLDER/multi-lingual/<model>/<language>/*results.csv

Outputs (saved to RESULTS_FOLDER/_results/statistics/)
-------------------------------------------------------
    wilcoxon_llm_vs_clinician.xlsx      — sheets: Overall, Anxiety, Mood, Stress
    mcnemar_llm_vs_llm.xlsx             — sheets: Overall, Anxiety, Mood, Stress
    non_inferiority_equivalence.xlsx    — sheets: Overall, Anxiety, Mood, Stress
    topn_vs_random.xlsx                 — sheets: By_Category, Overall
    multilingual_mcnemar.xlsx           — one sheet per model
"""

from pathlib import Path

import numpy as np
import pandas as pd

from icd11_utils import (
    LANGUAGES,
    apply_label_corrections,
    load_results,
    sort_models,
)
from stats_utils import (
    CATEGORIES,
    NI_MARGIN,
    apply_multiple_corrections,
    build_mcnemar_results,
    build_multilingual_mcnemar_results,
    build_ni_equivalence_results,
    build_topn_results,
    build_wilcoxon_results,
    load_clinician_mean_accuracy,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH      = Path("/Users/muellv01/Library/CloudStorage/OneDrive-NYULangoneHealth/Projects/ICD11_WHO")
RESULTS_FOLDER = BASE_PATH / "results_Apr26"

CLINICIAN_FILE = RESULTS_FOLDER / "_results" / "clinicians" / "clinicians_harmonised.csv"

OUTPUT_DIR = RESULTS_FOLDER / "_results" / "_statistics"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_excel(sheets: dict[str, pd.DataFrame], path: Path, index: bool = False) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=index)
    print(f"  Saved: {path}")


def normalize_llm_index_to_clinician(
    llm_index: pd.Index,
    clinician_index: pd.Index,
) -> pd.Index:
    """
    Truncate LLM Vignette_IDs to match the shorter clinician IDs via prefix match.

    Clinician IDs are sorted longest-first so longer prefixes take priority.
    LLM IDs with no prefix match are left unchanged and will be excluded
    from clinician-aligned tests via the intersection step downstream.
    """
    cli_ids = sorted(clinician_index.tolist(), key=len, reverse=True)
    mapping = {}
    for llm_id in llm_index:
        for cli_id in cli_ids:
            if str(llm_id).startswith(str(cli_id)):
                mapping[llm_id] = cli_id
                break
    return llm_index.map(lambda x: mapping.get(x, x))


def _build_correctness_wide(
    combined: pd.DataFrame,
    clinician_mean_acc: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Pivot combined LLM result DataFrame into a wide binary correctness table.

    Index  : MultiIndex(Category, Vignette_ID)
    Columns: one per LLM model
    Values : 1 if Predicted_Label == Ground_Truth_Label, else 0

    If clinician_mean_acc is provided, LLM Vignette_IDs are first normalised
    to match clinician IDs via prefix truncation (logged to stdout).
    The clinician column is NOT added here — it is handled separately in each
    test builder to keep the LLM-only correctness table clean for McNemar.
    """
    combined = combined.copy()
    combined.index.name = "Vignette_ID"

    # Normalise LLM vignette IDs to clinician prefix IDs
    if clinician_mean_acc is not None:
        normalised = normalize_llm_index_to_clinician(
            combined.index, clinician_mean_acc.index
        )
        combined.index = normalised
        n_matched = normalised.isin(clinician_mean_acc.index).sum()
        print(f"  ID normalisation: {n_matched}/{len(normalised)} LLM rows "
              f"matched to a clinician vignette ID.")

    combined["correct"] = (
        combined["Predicted_Label"] == combined["Ground_Truth_Label"]
    ).astype(int)

    rows = []
    for llm, grp in combined.groupby("llm"):
        sub = grp[["Category", "correct"]].copy()
        sub.columns = ["Category", llm]
        rows.append(sub)

    wide = rows[0].copy()
    for sub in rows[1:]:
        wide = wide.join(sub[[sub.columns[-1]]], how="outer")

    wide = wide.reset_index()
    wide = wide.rename(columns={"index": "Vignette_ID"}) \
               if "index" in wide.columns else wide
    wide = wide.set_index(["Category", "Vignette_ID"])
    return wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Load English LLM results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading English LLM results...")
    print("=" * 60)

    english_files = sorted(RESULTS_FOLDER.glob("english/*/english/*results.csv"))
    english_files = [f for f in english_files if "results_old" not in str(f)]
    print(f"  Found {len(english_files)} English result files.")

    combined_en = load_results(english_files, llm_part_index=-3)
    combined_en = apply_label_corrections(combined_en)
    combined_en.index.name = "Vignette_ID"

    # -----------------------------------------------------------------------
    # Load clinician mean accuracy
    # -----------------------------------------------------------------------
    clinician_mean_acc = None
    if CLINICIAN_FILE.exists():
        clinician_mean_acc = load_clinician_mean_accuracy(CLINICIAN_FILE)
        print(f"\n  Clinician mean accuracy loaded: "
              f"{len(clinician_mean_acc)} vignettes.")
        print(f"  Overall clinician mean accuracy: "
              f"{clinician_mean_acc.mean():.3f}")
    else:
        print(f"\n  WARNING: {CLINICIAN_FILE} not found. "
              f"Clinician comparison tests will be skipped.")

    # -----------------------------------------------------------------------
    # Build wide correctness table (LLMs only — clinician handled per test)
    # -----------------------------------------------------------------------
    correctness_en = _build_correctness_wide(combined_en, clinician_mean_acc)
    llm_cols = sort_models(list(correctness_en.columns))
    print(f"\n  Models: {llm_cols}")
    print(f"  Vignettes in correctness table: "
          f"{correctness_en.index.get_level_values('Vignette_ID').nunique()}")

    # -----------------------------------------------------------------------
    # Test 1: Wilcoxon — LLM vs. clinician mean accuracy
    # -----------------------------------------------------------------------
    if clinician_mean_acc is not None:
        print("\n[1/5] Wilcoxon signed-rank: LLM vs. clinician mean accuracy...")

        wilcoxon_results = build_wilcoxon_results(
            correctness_en,
            clinician_mean_acc,
            llm_cols=llm_cols,
            categories=CATEGORIES,
        )

        wilcoxon_sheets: dict[str, pd.DataFrame] = {}
        for scope in ["Overall"] + CATEGORIES:
            subset = wilcoxon_results[wilcoxon_results["Category"] == scope]
            if not subset.empty:
                wilcoxon_sheets[scope] = subset.reset_index(drop=True)

        _save_excel(
            wilcoxon_sheets,
            OUTPUT_DIR / "wilcoxon_llm_vs_clinician.xlsx",
        )
    else:
        print("\n[1/5] Skipping Wilcoxon test (no clinician data).")

    # -----------------------------------------------------------------------
    # Test 2: McNemar — LLM vs. LLM pairwise
    # -----------------------------------------------------------------------
    print("\n[2/5] McNemar: LLM vs. LLM pairwise...")

    mcnemar_llm = build_mcnemar_results(correctness_en, categories=CATEGORIES)

    llm_llm_sheets: dict[str, pd.DataFrame] = {}
    for scope in ["Overall"] + CATEGORIES:
        subset = mcnemar_llm[mcnemar_llm["Category"] == scope]
        if not subset.empty:
            llm_llm_sheets[scope] = subset.reset_index(drop=True)

    _save_excel(
        llm_llm_sheets,
        OUTPUT_DIR / "mcnemar_llm_vs_llm.xlsx",
    )

    # -----------------------------------------------------------------------
    # Test 3: Non-inferiority + Equivalence (TOST) — LLM vs. clinician
    # -----------------------------------------------------------------------
    if clinician_mean_acc is not None:
        print(f"\n[3/5] Non-inferiority + TOST equivalence (δ = {NI_MARGIN})...")

        ni_eq_results = build_ni_equivalence_results(
            correctness_en,
            clinician_mean_acc,
            llm_cols=llm_cols,
            margin=NI_MARGIN,
            categories=CATEGORIES,
        )

        ni_eq_sheets: dict[str, pd.DataFrame] = {}
        for scope in ["Overall"] + CATEGORIES:
            subset = ni_eq_results[ni_eq_results["Category"] == scope]
            if not subset.empty:
                ni_eq_sheets[scope] = subset.reset_index(drop=True)

        _save_excel(
            ni_eq_sheets,
            OUTPUT_DIR / "non_inferiority_equivalence.xlsx",
        )
    else:
        print("\n[3/5] Skipping NI + TOST tests (no clinician data).")

    # -----------------------------------------------------------------------
    # Test 4: Top-N vs. random baseline
    # -----------------------------------------------------------------------
    print("\n[4/5] Top-N vs. random baseline...")

    topn_files = sorted(RESULTS_FOLDER.glob("english/*/english/*results.csv"))
    topn_files = [f for f in topn_files if "results_old" not in str(f)]

    _probe = pd.read_csv(topn_files[0], nrows=1) if topn_files else pd.DataFrame()
    has_topn_cols = all(f"Top_{k}_Accuracy" in _probe.columns for k in [1, 2, 3])

    if has_topn_cols:
        topn_cat, topn_ovr = build_topn_results(topn_files, llm_part_index=-3)
        _save_excel(
            {"By_Category": topn_cat, "Overall": topn_ovr},
            OUTPUT_DIR / "topn_vs_random.xlsx",
        )
    else:
        print("  WARNING: Top_k_Accuracy columns not found in result files. "
              "Re-run build_topn_sheet() from icd11_utils to add them first.")

    # -----------------------------------------------------------------------
    # Test 5: Multilingual McNemar (per model, pairwise languages)
    # -----------------------------------------------------------------------
    print("\n[5/5] Multilingual McNemar (pairwise languages per model)...")

    lang_correctness: dict[str, pd.DataFrame] = {}

    for lang in LANGUAGES:
        if lang == "english":
            files = sorted(RESULTS_FOLDER.glob("english/*/english/*results.csv"))
        else:
            files = sorted(RESULTS_FOLDER.glob(f"multi-lingual/*/{lang}/*results.csv"))
        files = [f for f in files if "results_old" not in str(f)]

        if not files:
            print(f"  No files for {lang}, skipping.")
            continue

        combined_lang = load_results(files, llm_part_index=-3)
        combined_lang = apply_label_corrections(combined_lang)
        combined_lang.index.name = "Vignette_ID"
        combined_lang["correct"] = (
            combined_lang["Predicted_Label"] == combined_lang["Ground_Truth_Label"]
        ).astype(int)

        wide_lang = combined_lang.pivot_table(
            index=combined_lang.index,
            columns="llm",
            values="correct",
            aggfunc="first",
        )
        wide_lang.columns.name = None
        lang_correctness[lang] = wide_lang
        print(f"  {lang}: {len(wide_lang)} vignettes, "
              f"{wide_lang.shape[1]} models.")

    if len(lang_correctness) >= 2:
        multi_results = build_multilingual_mcnemar_results(
            lang_correctness,
            reference_lang="english",
        )
        multi_sheets: dict[str, pd.DataFrame] = {}
        for model, grp in multi_results.groupby("Model"):
            multi_sheets[str(model)[:31]] = grp.reset_index(drop=True)

        _save_excel(
            multi_sheets,
            OUTPUT_DIR / "multilingual_mcnemar.xlsx",
        )
    else:
        print("  Fewer than 2 languages loaded — skipping multilingual tests.")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("All statistical tests complete. Outputs in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)