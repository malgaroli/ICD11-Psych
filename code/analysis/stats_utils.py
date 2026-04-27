"""
stats_utils.py
==============
Shared statistical test functions for the ICD-11 LLM diagnostic benchmarking project.

Imported by:
    - run_statistics.py

Tests implemented
-----------------
1. McNemar test            — pairwise comparison of binary correctness vectors
                             (LLM vs. LLM only; LLM vs. clinician uses Wilcoxon)
2. Wilcoxon signed-rank    — per-vignette LLM correctness vs. clinician mean accuracy
3. Non-inferiority test    — one-sided proportion test on accuracy difference
                             with a pre-specified margin δ
4. Equivalence test (TOST) — two one-sided tests to establish equivalence within ±δ
5. Top-N vs. random        — one-sided proportions z-test per model per category,
                             plus marginal improvement tests (Top-k → Top-k+1)
6. Multilingual McNemar    — pairwise language comparisons per model
7. Multiple comparison     — Holm and Benjamini–Hochberg (FDR) corrections

Clinician handling
------------------
Clinicians are loaded from clinicians_harmonised.csv which contains one row per
clinician per vignette. Accuracy is computed as the mean proportion of clinicians
correct on each vignette (Ground_Truth_Label == Predicted_Label). This produces
a continuous [0, 1] vector per vignette, NOT a single binary vector. Therefore:

  - LLM vs. clinician comparison  : Wilcoxon signed-rank on per-vignette pairs
  - Non-inferiority / equivalence : proportion test using mean clinician accuracy
                                    as the reference proportion
  - McNemar                       : LLM vs. LLM only (requires paired binary data)
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


# ---------------------------------------------------------------------------
# McNemar test (LLM vs. LLM only)
# ---------------------------------------------------------------------------

def _mcnemar_one_pair(
    a: pd.Series,
    b: pd.Series,
) -> dict:
    """
    Run McNemar test on two binary (0/1) correctness vectors aligned by index.

    Uses exact binomial test when n_discordant < 25, chi-square with
    continuity correction otherwise.

    If n_discordant == 0 (identical outputs), returns NaN for statistic and
    p_value with a note — this is mathematically correct behaviour, not a bug.

    Returns a dict with: b01, b10, n_discordant, statistic, p_value, exact, note.
    """
    aligned = pd.concat([a, b], axis=1).dropna()
    aligned.columns = ["a", "b"]

    n11 = int(((aligned["a"] == 1) & (aligned["b"] == 1)).sum())
    n10 = int(((aligned["a"] == 1) & (aligned["b"] == 0)).sum())
    n01 = int(((aligned["a"] == 0) & (aligned["b"] == 1)).sum())
    n00 = int(((aligned["a"] == 0) & (aligned["b"] == 0)).sum())

    table = [[n11, n10], [n01, n00]]
    n_discordant = n01 + n10

    if n_discordant == 0:
        return {
            "b01": n01, "b10": n10, "n_discordant": 0,
            "statistic": np.nan, "p_value": np.nan,
            "exact": np.nan,
            "note": "Identical outputs — McNemar not applicable",
        }

    exact = n_discordant < 25
    result = sm_mcnemar(table, exact=exact, correction=not exact)

    return {
        "b01":          n01,
        "b10":          n10,
        "n_discordant": n_discordant,
        "statistic":    result.statistic,
        "p_value":      result.pvalue,
        "exact":        exact,
        "note":         "",
    }


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
# Wilcoxon signed-rank: LLM binary correctness vs. clinician mean accuracy
# ---------------------------------------------------------------------------

def run_wilcoxon_llm_vs_clinician(
    llm_correct: np.ndarray,
    clinician_mean: np.ndarray,
    model_name: str,
    category: str,
) -> dict:
    """
    Wilcoxon signed-rank test comparing per-vignette LLM correctness (0/1)
    against per-vignette clinician mean accuracy ([0, 1]).

    H0: the distribution of differences (LLM - clinician) is symmetric around 0.

    Parameters
    ----------
    llm_correct      : binary array, length = n_vignettes
    clinician_mean   : float array [0,1], length = n_vignettes (same order)
    model_name       : str label
    category         : str label

    Returns
    -------
    dict with: n, mean_llm, mean_clinician, mean_diff,
               statistic, p_value (two-sided),
               direction ("LLM > Clinician" / "LLM < Clinician" / "No difference")
    """
    a = np.asarray(llm_correct,    dtype=float)
    b = np.asarray(clinician_mean, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)

    if n == 0:
        return {
            "Model": model_name, "Category": category,
            **{k: np.nan for k in ["n", "mean_llm", "mean_clinician",
                                    "mean_diff", "statistic", "p_value", "direction"]},
        }

    diff = a - b

    if np.all(diff == 0):
        return {
            "Model": model_name, "Category": category,
            "n": n,
            "mean_llm":       round(float(a.mean()), 4),
            "mean_clinician": round(float(b.mean()), 4),
            "mean_diff":      0.0,
            "statistic":      np.nan,
            "p_value":        np.nan,
            "direction":      "No difference",
            "note":           "All differences zero — Wilcoxon not applicable",
        }

    stat, pval = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")

    direction = (
        "LLM > Clinician" if a.mean() > b.mean() else
        "LLM < Clinician" if a.mean() < b.mean() else
        "No difference"
    )
    # Hodges-Lehmann estimator: median of all pairwise averages of differences
    # Hodges-Lehmann estimator: median of all pairwise averages of differences
    hl_estimate = float(np.median(np.add.outer(diff, diff) / 2))

    return {
        "Model":          model_name,
        "Category":       category,
        "n":              n,
        "mean_llm":       round(float(a.mean()), 4),
        "mean_clinician": round(float(b.mean()), 4),
        "mean_diff":      round(float(diff.mean()), 4),
        "statistic":      round(float(stat), 4),
        "p_value":        round(float(pval), 6),
        "hl_estimate":    round(hl_estimate, 4),   # <-- new
        "direction":      direction,
        "note":           "",
    }


def build_wilcoxon_results(
    correctness_wide: pd.DataFrame,
    clinician_mean_acc: pd.Series,
    llm_cols: list[str],
    categories: list[str] = CATEGORIES,
) -> pd.DataFrame:
    """
    Run Wilcoxon signed-rank tests for each LLM vs. clinician mean accuracy,
    overall and per category.

    Parameters
    ----------
    correctness_wide   : MultiIndex(Category, Vignette_ID) × LLM columns (0/1)
    clinician_mean_acc : Series indexed by Vignette_ID, values in [0,1]
    llm_cols           : list of LLM column names to test

    Returns
    -------
    DataFrame with one row per (model, category), Holm/FDR corrections applied.
    """
    all_results = []

    def _run(df_group: pd.DataFrame, cat_label: str) -> None:
        # Vignette IDs are the index level after dropping Category
        vignette_ids = df_group.index.get_level_values("Vignette_ID") \
            if "Vignette_ID" in df_group.index.names \
            else df_group.index
        shared = vignette_ids.intersection(clinician_mean_acc.index)
        if len(shared) == 0:
            return
        cli_vals = clinician_mean_acc.loc[shared].values
        for llm in llm_cols:
            if llm not in df_group.columns:
                continue
            llm_vals = df_group.loc[
                df_group.index.isin(shared) if df_group.index.nlevels == 1
                else df_group.index.get_level_values("Vignette_ID").isin(shared),
                llm,
            ].values
            # Align lengths defensively
            min_len = min(len(llm_vals), len(cli_vals))
            r = run_wilcoxon_llm_vs_clinician(
                llm_vals[:min_len], cli_vals[:min_len], llm, cat_label
            )
            all_results.append(r)

    # Per category
    for cat in categories:
        if cat not in correctness_wide.index.get_level_values("Category"):
            continue
        df_cat = correctness_wide.xs(cat, level="Category")
        # Restore Vignette_ID in index for alignment
        df_cat.index.name = "Vignette_ID"
        _run(df_cat, cat)

    # Overall — reset to Vignette_ID index
    df_all = correctness_wide.copy()
    df_all = df_all.reset_index(level="Category", drop=True)
    df_all.index.name = "Vignette_ID"
    _run(df_all, "Overall")

    df = pd.DataFrame(all_results)
    df = apply_multiple_corrections(df)
    return df


# ---------------------------------------------------------------------------
# Non-inferiority test (LLM vs. clinician mean accuracy)
# ---------------------------------------------------------------------------

def run_non_inferiority_test(
    llm_correct: np.ndarray,
    clinician_mean: np.ndarray,
    margin: float = NI_MARGIN,
    alpha: float = ALPHA,
) -> dict:
    """
    One-sided non-inferiority test: H0: p_llm - p_clinician <= -margin
    H1: LLM is not worse than clinician by more than margin.

    Uses the paired McNemar-based SE on the difference.

    Note: clinician_mean is a per-vignette proportion (mean of binary clinician
    responses). We binarise it at 0.5 for the paired SE calculation, but use
    the raw proportions for the point estimates and confidence intervals.

    Returns a dict with:
        n, p_llm, p_clinician, diff, SE_diff,
        z_ni, p_ni (one-sided),
        CI95_lower, CI95_upper,
        NI_lower_95,
        non_inferior (bool)
    """
    a = np.asarray(llm_correct, dtype=float)
    b = np.asarray(clinician_mean, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)

    if n == 0:
        return {k: np.nan for k in [
            "n", "p_llm", "p_clinician", "diff", "SE_diff",
            "z_ni", "p_ni","hl_estimate", "CI95_lower", "CI95_upper",
            "NI_lower_95", "NI_margin", "non_inferior",
        ]}

    p_llm  = float(a.mean())
    p_clin = float(b.mean())
    diff   = p_llm - p_clin

    # Paired SE: binarise clinician mean at 0.5 for discordant-pair calculation
    b_bin = (b >= 0.5).astype(float)
    b10 = float(((a == 1) & (b_bin == 0)).sum())
    b01 = float(((a == 0) & (b_bin == 1)).sum())
    se = np.sqrt((b01 + b10) / n**2 - (b10 - b01)**2 / n**3)

    if se == 0 or np.isnan(se):
        z_ni = np.nan
        p_ni = np.nan
    else:
        z_ni = (diff + margin) / se
        p_ni = float(stats.norm.sf(z_ni))

    z95   = 1.96
    ci_lo = diff - z95 * se
    ci_hi = diff + z95 * se
    ni_lb = diff - 1.645 * se   # one-sided 95% lower bound

    # Hodges-Lehmann estimator: median of all pairwise averages of differences
    hl_estimate = float(np.median(np.add.outer(diff, diff) / 2))


    return {
        "n":           n,
        "p_llm":       round(p_llm,  4),
        "p_clinician": round(p_clin, 4),
        "diff":        round(diff,   4),
        "SE_diff":     round(se,     6),
        "z_ni":        round(z_ni,   4) if not np.isnan(z_ni) else np.nan,
        "p_ni":        round(p_ni,   6) if not np.isnan(p_ni) else np.nan,
        "hl_estimate":    round(hl_estimate, 4),   # <-- new
        "CI95_lower":  round(ci_lo,  4),
        "CI95_upper":  round(ci_hi,  4),
        "NI_lower_95": round(ni_lb,  4),
        "NI_margin":   margin,
        "non_inferior": bool(ni_lb > -margin) if not np.isnan(ni_lb) else False,
    }


# ---------------------------------------------------------------------------
# Equivalence test (TOST — Two One-Sided Tests)
# ---------------------------------------------------------------------------

def run_equivalence_test(
    llm_correct: np.ndarray,
    clinician_mean: np.ndarray,
    margin: float = NI_MARGIN,
    alpha: float = ALPHA,
) -> dict:
    """
    TOST equivalence test: LLM performance is equivalent to clinician if
    the difference falls within (-margin, +margin).

    Two one-sided tests:
        H0_lower: diff <= -margin   (rejected by NI test)
        H0_upper: diff >= +margin   (rejected by superiority test)

    Equivalence is declared when BOTH H0_lower and H0_upper are rejected
    at level alpha (i.e., both p-values < alpha).

    Uses same paired SE as non-inferiority test.

    Returns a dict with:
        n, p_llm, p_clinician, diff, SE_diff,
        z_lower, p_lower,   (H0: diff <= -margin)
        z_upper, p_upper,   (H0: diff >= +margin)
        p_tost              (max of p_lower, p_upper — the binding test)
        CI90_lower, CI90_upper  (90% CI — the standard for TOST at alpha=0.05)
        equivalent (bool: p_tost < alpha AND CI90 within [-margin, +margin])
    """
    a = np.asarray(llm_correct, dtype=float)
    b = np.asarray(clinician_mean, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)

    if n == 0:
        return {k: np.nan for k in [
            "n", "p_llm", "p_clinician", "diff", "SE_diff",
            "z_lower", "p_lower", "z_upper", "p_upper",
            "p_tost", "CI90_lower", "CI90_upper",
            "EQ_margin", "equivalent",
        ]}

    p_llm  = float(a.mean())
    p_clin = float(b.mean())
    diff   = p_llm - p_clin

    b_bin = (b >= 0.5).astype(float)
    b10 = float(((a == 1) & (b_bin == 0)).sum())
    b01 = float(((a == 0) & (b_bin == 1)).sum())
    se = np.sqrt((b01 + b10) / n**2 - (b10 - b01)**2 / n**3)

    if se == 0 or np.isnan(se):
        return {
            "n": n, "p_llm": round(p_llm, 4), "p_clinician": round(p_clin, 4),
            "diff": round(diff, 4), "SE_diff": np.nan,
            "z_lower": np.nan, "p_lower": np.nan,
            "z_upper": np.nan, "p_upper": np.nan,
            "p_tost": np.nan, "CI90_lower": np.nan, "CI90_upper": np.nan,
            "EQ_margin": margin, "equivalent": False,
        }

    # Lower one-sided test: H0: diff <= -margin  →  z = (diff - (-margin)) / se
    z_lower = (diff + margin) / se
    p_lower = float(stats.norm.sf(z_lower))   # upper tail

    # Upper one-sided test: H0: diff >= +margin  →  z = (diff - margin) / se
    z_upper = (diff - margin) / se
    p_upper = float(stats.norm.cdf(z_upper))  # lower tail

    p_tost = max(p_lower, p_upper)

    # 90% CI (corresponds to alpha=0.05 TOST)
    z90    = 1.645
    ci90_lo = diff - z90 * se
    ci90_hi = diff + z90 * se

    equivalent = bool(
        p_tost < alpha
        and ci90_lo > -margin
        and ci90_hi < margin
    )

    # Hodges-Lehmann estimator: median of all pairwise averages of differences
    hl_estimate = float(np.median(np.add.outer(diff, diff) / 2))

    return {
        "n":           n,
        "p_llm":       round(p_llm,   4),
        "p_clinician": round(p_clin,  4),
        "diff":        round(diff,    4),
        "SE_diff":     round(se,      6),
        "z_lower":     round(z_lower, 4),
        "p_lower":     round(p_lower, 6),
        "z_upper":     round(z_upper, 4),
        "p_upper":     round(p_upper, 6),
        "p_tost":      round(p_tost,  6),
        "hl_estimate":    round(hl_estimate, 4),   # <-- new
        "CI90_lower":  round(ci90_lo, 4),
        "CI90_upper":  round(ci90_hi, 4),
        "EQ_margin":   margin,
        "equivalent":  equivalent,
    }


def build_ni_equivalence_results(
    correctness_wide: pd.DataFrame,
    clinician_mean_acc: pd.Series,
    llm_cols: list[str],
    margin: float = NI_MARGIN,
    categories: list[str] = CATEGORIES,
) -> pd.DataFrame:
    """
    Run both non-inferiority and equivalence (TOST) tests for each LLM
    vs. clinician mean accuracy, overall and per category.

    Returns a single DataFrame with NI and TOST columns side by side.
    """
    all_results = []

    def _run(vignette_ids: pd.Index, cat_label: str) -> None:
        shared = vignette_ids.intersection(clinician_mean_acc.index)
        if len(shared) == 0:
            return
        cli_vals = clinician_mean_acc.loc[shared].values

        for llm in llm_cols:
            # Extract LLM correctness aligned to shared vignette IDs
            if correctness_wide.index.nlevels == 2:
                try:
                    if cat_label == "Overall":
                        llm_vals = correctness_wide[llm].loc[
                            correctness_wide.index.get_level_values("Vignette_ID").isin(shared)
                        ].values
                    else:
                        df_cat = correctness_wide.xs(cat_label, level="Category")
                        llm_vals = df_cat.loc[shared, llm].values
                except KeyError:
                    continue
            else:
                llm_vals = correctness_wide.loc[shared, llm].values

            min_len = min(len(llm_vals), len(cli_vals))
            lv = llm_vals[:min_len]
            cv = cli_vals[:min_len]

            ni  = run_non_inferiority_test(lv, cv, margin=margin)
            eq  = run_equivalence_test(lv, cv, margin=margin)

            combined = {
                "Model":    llm,
                "Category": cat_label,
                # NI columns
                "n":              ni["n"],
                "p_llm":          ni["p_llm"],
                "p_clinician":    ni["p_clinician"],
                "diff":           ni["diff"],
                "SE_diff":        ni["SE_diff"],
                "z_ni":           ni["z_ni"],
                "p_ni":           ni["p_ni"],
                "CI95_lower":     ni["CI95_lower"],
                "CI95_upper":     ni["CI95_upper"],
                "NI_lower_95":    ni["NI_lower_95"],
                "NI_margin":      ni["NI_margin"],
                "non_inferior":   ni["non_inferior"],
                "hl_estimate":    ni["hl_estimate"],  
                # TOST columns
                "z_lower":        eq["z_lower"],
                "p_lower":        eq["p_lower"],
                "z_upper":        eq["z_upper"],
                "p_upper":        eq["p_upper"],
                "p_tost":         eq["p_tost"],
                "CI90_lower":     eq["CI90_lower"],
                "CI90_upper":     eq["CI90_upper"],
                "EQ_margin":      eq["EQ_margin"],
                "equivalent":     eq["equivalent"],
            }
            all_results.append(combined)

    # Per category
    for cat in categories:
        if cat not in correctness_wide.index.get_level_values("Category"):
            continue
        df_cat = correctness_wide.xs(cat, level="Category")
        df_cat.index.name = "Vignette_ID"
        _run(df_cat.index, cat)

    # Overall
    all_vignette_ids = correctness_wide.index.get_level_values("Vignette_ID")
    _run(pd.Index(all_vignette_ids), "Overall")

    return pd.DataFrame(all_results).sort_values(["Category", "Model"]).reset_index(drop=True)


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