"""
stats_utils.py
==============
Shared statistical test functions for the ICD-11 LLM diagnostic benchmarking project.

Imported by:
    - run_statistics.py

Tests implemented
-----------------
1. Top-N vs. random        — one-sided proportions z-test per model per category,
                             plus marginal improvement tests (Top-k → Top-k+1)
2. Multilingual McNemar    — pairwise language comparisons per model
3. Multiple comparison     — Holm and Benjamini–Hochberg (FDR) corrections

Clinician handling
------------------
Clinicians are loaded from clinicians_harmonised.csv which contains one row per
clinician per vignette. Accuracy is computed as the mean proportion of clinicians
correct on each vignette (Ground_Truth_Label == Predicted_Label). This produces
a continuous [0, 1] vector per vignette, NOT a single binary vector.

"""

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["Anxiety", "Mood", "Stress"]

# Number of answer choices per category — used as the random-chance denominator
N_CHOICES = {"Mood": 13, "Stress": 8, "Anxiety": 16}
N_CHOICES_OVERALL = int(np.mean(list(N_CHOICES.values())))  # ≈ 12

# Non-inferiority / equivalence margin (absolute proportion units)
NI_MARGIN = 0.10   # δ = 10 percentage points

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------

def apply_multiple_corrections(df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Add Holm and FDR (Benjamini–Hochberg) corrected p-values to a DataFrame.

    Adds columns:
        p_holm, p_fdr, significant_holm, significant_fdr, Correction_Significant
    """
    df = df.copy()
    if df.empty or "p_value" not in df.columns:
        return df

    valid = df["p_value"].notna()
    if valid.sum() == 0:
        return df

    df.loc[valid, "p_holm"] = multipletests(df.loc[valid, "p_value"], method="holm")[1]
    df.loc[valid, "p_fdr"]  = multipletests(df.loc[valid, "p_value"], method="fdr_bh")[1]

    df["significant_holm"] = df["p_holm"] < alpha
    df["significant_fdr"]  = df["p_fdr"]  < alpha
    df["Correction_Significant"] = np.select(
        [df["significant_holm"], df["significant_fdr"]],
        ["Holm", "FDR"],
        default="None",
    )
    return df


# ---------------------------------------------------------------------------
# Clinician mean accuracy loader
# ---------------------------------------------------------------------------

def load_clinician_mean_accuracy(path: Path) -> pd.Series:
    """
    Load clinicians_harmonised.csv and compute per-vignette mean accuracy.

    The file contains one row per clinician per vignette with columns:
        Vignette_ID, ID, Category, Language, Ground_Truth_Label, Predicted_Label

    Returns
    -------
    pd.Series
        Index = Vignette_ID (str)
        Values = mean proportion of clinicians correct on that vignette [0.0, 1.0]
        Name = "Clinician_Mean_Accuracy"

    Also returns a helper DataFrame with Category per vignette for downstream
    category-level subsetting — accessible as the .attrs["category_map"] attribute.
    """
    df = pd.read_csv(path)
    df["Vignette_ID"]    = df["Vignette_ID"].astype(str).str.strip()
    df["correct"]        = (
        df["Predicted_Label"].str.strip() == df["Ground_Truth_Label"].str.strip()
    ).astype(int)

    # Mean accuracy per vignette
    mean_acc = df.groupby("Vignette_ID")["correct"].mean()
    mean_acc.name = "Clinician_Mean_Accuracy"

    # Category map — one category per vignette (take first occurrence)
    cat_map = df.groupby("Vignette_ID")["Category"].first()
    mean_acc.attrs["category_map"] = cat_map

    return mean_acc

def run_mcnemar_pairwise(
    correctness_wide: pd.DataFrame,
    category_label: str,
) -> list[dict]:
    """
    Run all pairwise McNemar tests across columns (LLM models only).

    Parameters
    ----------
    correctness_wide : pd.DataFrame
        Columns = model names; rows = vignettes; values = 0/1 correctness.
        Do NOT include the clinician column here.
    category_label : str
        Label for the 'Category' column in the output.

    Returns
    -------
    List of result dicts, one per pair.
    """
    results = []
    for col_a, col_b in combinations(correctness_wide.columns, 2):
        pair_result = _mcnemar_one_pair(
            correctness_wide[col_a],
            correctness_wide[col_b],
        )
        results.append({
            "Category": category_label,
            "Model_A":  col_a,
            "Model_B":  col_b,
            **pair_result,
        })
    return results


def build_mcnemar_results(
    correctness_wide: pd.DataFrame,
    categories: list[str] = CATEGORIES,
) -> pd.DataFrame:
    """
    Run pairwise McNemar tests overall and per category (LLM vs. LLM only).

    Parameters
    ----------
    correctness_wide : pd.DataFrame
        MultiIndex (Category, Vignette_ID) × LLM columns; values = 0/1.
        Must NOT include a clinician column.

    Returns
    -------
    DataFrame with all results, Holm/FDR corrections applied.
    """
    all_results = []

    for cat in categories:
        if cat not in correctness_wide.index.get_level_values("Category"):
            continue
        df_cat = correctness_wide.xs(cat, level="Category", drop_level=True)
        all_results.extend(run_mcnemar_pairwise(df_cat, cat))

    df_overall = correctness_wide.reset_index(drop=True)
    all_results.extend(run_mcnemar_pairwise(df_overall, "Overall"))

    df = pd.DataFrame(all_results)
    df = apply_multiple_corrections(df)
    return df

# ---------------------------------------------------------------------------
# Top-N vs. random baseline
# ---------------------------------------------------------------------------

def run_topn_vs_random(
    df: pd.DataFrame,
    n_diagnoses: int,
    model_name: str,
    category: str,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    One-sided proportions z-test (H1: model > chance) for Top-1, Top-2, Top-3,
    plus marginal improvement tests (Top-1→Top-2, Top-2→Top-3).
    """
    N = len(df)
    results = []

    counts   = {k: int(df[f"Top_{k}_Accuracy"].sum())  for k in [1, 2, 3]}
    accs     = {k: df[f"Top_{k}_Accuracy"].mean()       for k in [1, 2, 3]}
    expected = {k: k / n_diagnoses                      for k in [1, 2, 3]}

    for k in [1, 2, 3]:
        count = counts[k]
        stat, pval = proportions_ztest(count, N, value=expected[k], alternative="larger")
        ci_lo, ci_hi = proportion_confint(count, N, alpha=alpha, method="wilson")

        results.append({
            "Model":             model_name,
            "Category":          category,
            "Test_Type":         f"Top-{k} vs. random",
            "Observed_Accuracy": round(accs[k], 4),
            "Correct_Count":     count,
            "N":                 N,
            "Expected_Random":   round(expected[k], 4),
            "n_choices":         n_diagnoses,
            "z_value":           round(stat, 4),
            "p_value":           round(pval, 6),
            "CI_Lower":          round(ci_lo, 4),
            "CI_Upper":          round(ci_hi, 4),
        })

    for a_k, b_k in [(1, 2), (2, 3)]:
        improved   = (df[f"Top_{b_k}_Accuracy"].astype(bool) & ~df[f"Top_{a_k}_Accuracy"].astype(bool))
        n_improved = int(improved.sum())
        p_improved = n_improved / N
        p_exp      = (b_k - a_k) / n_diagnoses

        stat, pval = proportions_ztest(n_improved, N, value=p_exp, alternative="larger")
        ci_lo, ci_hi = proportion_confint(n_improved, N, alpha=alpha, method="wilson")

        results.append({
            "Model":             model_name,
            "Category":          category,
            "Test_Type":         f"Improvement Top-{a_k}→Top-{b_k}",
            "Observed_Accuracy": round(p_improved, 4),
            "Correct_Count":     n_improved,
            "N":                 N,
            "Expected_Random":   round(p_exp, 4),
            "n_choices":         n_diagnoses,
            "z_value":           round(stat, 4),
            "p_value":           round(pval, 6),
            "CI_Lower":          round(ci_lo, 4),
            "CI_Upper":          round(ci_hi, 4),
        })

    return pd.DataFrame(results)


def build_topn_results(
    files: list[Path],
    llm_part_index: int = -3,
    n_choices: dict = N_CHOICES,
    categories: list[str] = CATEGORIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loop over per-model result files, run top-N tests per category and overall,
    apply Holm/FDR corrections, and return (by_category_df, overall_df).
    """
    all_cat_results = []
    all_ovr_results = []
    n_overall = int(np.mean(list(n_choices.values())))

    for file in files:
        model = file.parts[llm_part_index]
        df = pd.read_csv(file, index_col=["Vignette_ID", "Category"])

        for cat in categories:
            if cat not in df.index.get_level_values("Category"):
                continue
            df_cat = df.xs(cat, level="Category")[
                ["Top_1_Accuracy", "Top_2_Accuracy", "Top_3_Accuracy"]
            ]
            res = run_topn_vs_random(df_cat, n_choices[cat], model, cat)
            all_cat_results.append(res)

        df_all = df[["Top_1_Accuracy", "Top_2_Accuracy", "Top_3_Accuracy"]]
        res_ovr = run_topn_vs_random(df_all, n_overall, model, "Overall")
        all_ovr_results.append(res_ovr)

    df_cat = pd.concat(all_cat_results, ignore_index=True)
    df_ovr = pd.concat(all_ovr_results, ignore_index=True)

    df_cat = apply_multiple_corrections(df_cat)
    df_ovr = apply_multiple_corrections(df_ovr)

    return df_cat, df_ovr


# ---------------------------------------------------------------------------
# Multilingual McNemar (pairwise language comparisons, per model)
# ---------------------------------------------------------------------------

def build_multilingual_mcnemar_results(
    lang_correctness: dict[str, pd.DataFrame],
    reference_lang: str = "english",
) -> pd.DataFrame:
    """
    For each model, run pairwise McNemar tests between all language pairs.

    Parameters
    ----------
    lang_correctness : dict[str, pd.DataFrame]
        language → DataFrame with Vignette_ID index, model columns, 0/1 values.

    Returns
    -------
    DataFrame with Holm/FDR corrections applied per model.
    """
    languages  = list(lang_correctness.keys())
    all_model_sets = [set(df.columns) for df in lang_correctness.values()]
    models     = sorted(set.intersection(*all_model_sets))
    all_results = []

    for model in models:
        for lang_a, lang_b in combinations(languages, 2):
            df_a = lang_correctness[lang_a][[model]].rename(columns={model: lang_a})
            df_b = lang_correctness[lang_b][[model]].rename(columns={model: lang_b})
            merged = df_a.join(df_b, how="inner")
            if merged.empty:
                continue
            pair_result = _mcnemar_one_pair(merged[lang_a], merged[lang_b])
            all_results.append({
                "Model":  model,
                "Lang_A": lang_a,
                "Lang_B": lang_b,
                **pair_result,
            })

    df = pd.DataFrame(all_results)
    corrected_parts = []
    for model, grp in df.groupby("Model"):
        corrected_parts.append(apply_multiple_corrections(grp))
    return pd.concat(corrected_parts, ignore_index=True)
