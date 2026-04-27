"""
run_interrater_agreement.py
============================
Compute inter-rater agreement for clinician diagnostic vignette ratings
using two complementary metrics:

    1. Krippendorff's Alpha (nominal)
       - Primary metric. Handles variable rater counts per vignette via
         np.nan padding, makes no assumptions about label distribution,
         and is well-established in both clinical and NLP annotation work.
       - Computed once over the full dataset (and per category) treating
         all vignettes jointly — this is the statistically appropriate
         use of Krippendorff's formula.
       - Bootstrap 95% CI (n=1000 resamples over vignettes).

    2. Brennan-Prediger Kappa (nominal)
       - Fixes Kappa's Paradox by replacing the empirical chance baseline
         with a uniform 1/k baseline (k = number of possible categories).
         This makes it insensitive to label prevalence skew, unlike
         Fleiss' kappa.
       - Computed per vignette (using the global k for that category),
         then summarised as mean ± SD with a 95% t-based CI.
       - Per-vignette values are saved for inspection.

Both metrics are reported alongside percent agreement (already available
in clinician_classification_metrics.xlsx) for a complete picture.

Why not Fleiss' kappa?
    With ~49 raters per vignette and one dominant label per vignette,
    Fleiss' kappa produces near-zero or negative values because the
    empirical expected agreement (P_e) is itself very high — a known
    artefact called Kappa's Paradox (Feinstein & Cicchetti 1990).
    Brennan-Prediger corrects this; Krippendorff avoids it by design.

Input:
    results_Apr26/_results/clinicians/clinicians_harmonised.csv
        Columns: Vignette_ID (index), ID, Category,
                 Ground_Truth_Label, Predicted_Label

Outputs (saved to results_Apr26/_results/clinicians/):
    - clinician_interrater_agreement.xlsx
        Sheet "Summary"              — both metrics, overall + per category
        Sheet "BP_Per_Vignette"      — per-vignette Brennan-Prediger detail
        Sheet "BP_Per_Vignette_<Cat>"— per category
    - clinician_classification_metrics.xlsx  (updated in-place)
        Sheet "InterRater_Agreement" — same summary table appended

Usage:
    python run_interrater_agreement.py
"""

from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats

from icd11_utils import save_excel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH = Path(
    "/Users/muellv01/Library/CloudStorage/OneDrive-NYULangoneHealth/Projects/ICD11_WHO"
)
RESULTS_FOLDER = BASE_PATH / "results_Apr26"
OUTPUT_DIR = RESULTS_FOLDER / "_results" / "clinicians"

HARMONISED_CSV = OUTPUT_DIR / "clinicians_harmonised.csv"
METRICS_XLSX   = OUTPUT_DIR / "clinician_classification_metrics.xlsx"
OUTPUT_XLSX    = OUTPUT_DIR / "clinician_interrater_agreement.xlsx"

# Number of bootstrap resamples for Krippendorff CI
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Krippendorff's Alpha
# ---------------------------------------------------------------------------

def build_krippendorff_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build the (n_raters x n_vignettes) matrix required by the krippendorff
    library, with np.nan for missing ratings (raters who did not rate a
    given vignette).

    Labels are encoded as integers (1-based) for the nominal metric.
    The encoding is consistent within the matrix so ordinal distances
    are not accidentally introduced.

    Returns:
        matrix      : float array of shape (max_raters, n_vignettes)
        vignette_ids: list of vignette IDs (column order)
    """
    # Encode labels to integers — consistent across the full dataset
    all_labels = sorted(df["Predicted_Label"].unique())
    label_to_int = {lbl: i + 1 for i, lbl in enumerate(all_labels)}

    vignette_ids = sorted(df.index.unique())
    all_rater_ids = sorted(df["ID"].unique())

    n_raters   = len(all_rater_ids)
    n_vignettes = len(vignette_ids)

    rater_to_row    = {r: i for i, r in enumerate(all_rater_ids)}
    vignette_to_col = {v: j for j, v in enumerate(vignette_ids)}

    matrix = np.full((n_raters, n_vignettes), np.nan)

    for _, row in df.reset_index().iterrows():
        r = rater_to_row[row["ID"]]
        c = vignette_to_col[row["Vignette_ID"]]
        matrix[r, c] = label_to_int[row["Predicted_Label"]]

    return matrix, vignette_ids


def compute_krippendorff_alpha(
    matrix: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Compute Krippendorff's alpha (nominal) with bootstrap 95% CI.

    Bootstrap resamples vignettes (columns) with replacement so the CI
    reflects uncertainty over the vignette sample.

    Returns dict with keys: alpha, ci_lower, ci_upper, ci, n_vignettes,
    n_raters, mean_raters_per_vignette.
    """
    n_raters, n_vignettes = matrix.shape

    # Point estimate
    alpha_val = krippendorff.alpha(
        reliability_data=matrix,
        level_of_measurement="nominal",
    )

    # Bootstrap CI — resample columns (vignettes)
    rng = np.random.default_rng(seed)
    boot_alphas = []
    for _ in range(n_bootstrap):
        col_idx = rng.integers(0, n_vignettes, size=n_vignettes)
        boot_mat = matrix[:, col_idx]
        # Skip degenerate resamples where all values are identical or all NaN
        vals = boot_mat[~np.isnan(boot_mat)]
        if len(vals) == 0 or len(np.unique(vals)) < 2:
            continue
        try:
            a = krippendorff.alpha(
                reliability_data=boot_mat,
                level_of_measurement="nominal",
            )
            boot_alphas.append(a)
        except Exception:
            continue

    boot_alphas = np.array(boot_alphas)
    ci_lower = float(np.percentile(boot_alphas, 2.5))
    ci_upper = float(np.percentile(boot_alphas, 97.5))

    # Mean raters per vignette (non-NaN entries per column)
    ratings_per_vignette = (~np.isnan(matrix)).sum(axis=0)

    return {
        "alpha": round(float(alpha_val), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        "n_vignettes": n_vignettes,
        "n_raters": n_raters,
        "mean_raters_per_vignette": round(float(ratings_per_vignette.mean()), 1),
        "n_bootstrap": len(boot_alphas),
    }


# ---------------------------------------------------------------------------
# Brennan-Prediger Kappa
# ---------------------------------------------------------------------------

def _bp_kappa_single(vote_counts: np.ndarray, k_categories: int) -> float:
    """
    Brennan-Prediger kappa for one vignette.

    P_o = sum_j [n_j * (n_j - 1)] / [n * (n - 1)]   (observed pairwise agreement)
    P_e = 1 / k                                        (uniform chance baseline)
    BP  = (P_o - P_e) / (1 - P_e)

    Args:
        vote_counts  : 1-D integer array of counts per label (non-zero only)
        k_categories : total number of possible labels for this category
                       (global count, not just those seen in this vignette)
    """
    n = int(vote_counts.sum())
    if n < 2:
        return np.nan
    P_o = float(np.sum(vote_counts * (vote_counts - 1))) / (n * (n - 1))
    P_e = 1.0 / k_categories
    return (P_o - P_e) / (1.0 - P_e)


def compute_bp_per_vignette(df: pd.DataFrame, k_categories: int) -> pd.DataFrame:
    """
    Compute Brennan-Prediger kappa for every vignette in df.

    k_categories is the number of distinct possible labels for this subset
    (computed from the full subset before calling this function).

    Returns a DataFrame indexed by Vignette_ID with columns:
        Category, Ground_Truth_Label, n_raters, n_unique_labels,
        pct_agree, bp_kappa
    """
    records = []
    for vignette_id, vgroup in df.groupby(df.index):
        gt       = vgroup["Ground_Truth_Label"].iloc[0]
        category = vgroup["Category"].iloc[0]
        vote_counts = vgroup["Predicted_Label"].value_counts()
        n_raters = len(vgroup)
        n_unique = len(vote_counts)
        pct_agree = round(float(vote_counts.iloc[0]) / n_raters, 4)

        counts_arr = vote_counts.values.astype(int)
        bp = _bp_kappa_single(counts_arr, k_categories)

        records.append({
            "Vignette_ID":       vignette_id,
            "Category":          category,
            "Ground_Truth_Label":gt,
            "n_raters":          n_raters,
            "n_unique_labels":   n_unique,
            "pct_agree":         pct_agree,
            "bp_kappa":          round(float(bp), 4) if not np.isnan(bp) else np.nan,
        })

    return pd.DataFrame(records).set_index("Vignette_ID")


def summarise_bp(per_vignette_df: pd.DataFrame, subset_name: str) -> dict:
    """
    Summarise per-vignette Brennan-Prediger kappa as mean ± SD with
    95% t-based CI.  NaN vignettes (single rater) are excluded.
    """
    kappas = per_vignette_df["bp_kappa"].dropna().values
    n = len(kappas)
    n_excluded = int(per_vignette_df["bp_kappa"].isna().sum())

    if n == 0:
        return {
            "Subset": subset_name,
            "n_vignettes": 0,
            "n_excluded": n_excluded,
            "bp_mean": np.nan, "bp_std": np.nan, "bp_median": np.nan,
            "bp_ci_lower": np.nan, "bp_ci_upper": np.nan, "bp_ci": "N/A",
        }

    mean_k   = float(np.mean(kappas))
    std_k    = float(np.std(kappas, ddof=1))
    median_k = float(np.median(kappas))
    se       = std_k / np.sqrt(n)
    t_crit   = stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
    ci_lower = mean_k - t_crit * se
    ci_upper = mean_k + t_crit * se

    return {
        "Subset":       subset_name,
        "n_vignettes":  n,
        "n_excluded":   n_excluded,
        "bp_mean":      round(mean_k, 4),
        "bp_std":       round(std_k, 4),
        "bp_median":    round(median_k, 4),
        "bp_ci_lower":  round(ci_lower, 4),
        "bp_ci_upper":  round(ci_upper, 4),
        "bp_ci":        f"[{ci_lower:.4f}, {ci_upper:.4f}]",
    }


# ---------------------------------------------------------------------------
# Append to existing metrics workbook
# ---------------------------------------------------------------------------

def append_to_metrics(summary_df: pd.DataFrame, metrics_path: Path) -> None:
    if not metrics_path.exists():
        print(f"  Metrics workbook not found, skipping append: {metrics_path}")
        return
    with pd.ExcelWriter(
        metrics_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        summary_df.to_excel(writer, sheet_name="InterRater_Agreement", index=False)
    print(f"  Appended 'InterRater_Agreement' sheet to: {metrics_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading harmonised clinician ratings from:\n  {HARMONISED_CSV}\n")
    df = pd.read_csv(HARMONISED_CSV, index_col="Vignette_ID")
    df.columns = df.columns.str.strip()

    n_clinicians = df["ID"].nunique()
    n_vignettes  = df.index.nunique()
    categories   = sorted(df["Category"].unique())

    print(f"Clinicians:       {n_clinicians}")
    print(f"Unique vignettes: {n_vignettes}")
    print(f"Total ratings:    {len(df)}")
    print(f"Categories:       {categories}\n")

    # ------------------------------------------------------------------
    # 1. Krippendorff's Alpha
    # ------------------------------------------------------------------
    print("Computing Krippendorff's alpha...")

    # Overall
    matrix_overall, _ = build_krippendorff_matrix(df)
    ka_overall = compute_krippendorff_alpha(matrix_overall)
    ka_overall["Subset"] = "Overall"

    # Per category
    ka_by_cat = {}
    for cat in categories:
        cat_df = df[df["Category"] == cat]
        mat, _ = build_krippendorff_matrix(cat_df)
        ka = compute_krippendorff_alpha(mat)
        ka["Subset"] = cat
        ka_by_cat[cat] = ka

    # ------------------------------------------------------------------
    # 2. Brennan-Prediger Kappa
    # ------------------------------------------------------------------
    print("Computing Brennan-Prediger kappa per vignette...")

    # k = number of distinct labels in the subset (global for that scope)
    k_overall = df["Predicted_Label"].nunique()
    bp_overall_pv = compute_bp_per_vignette(df, k_categories=k_overall)
    bp_overall_summary = summarise_bp(bp_overall_pv, "Overall")

    bp_by_cat_pv: dict[str, pd.DataFrame] = {}
    bp_by_cat_summary: dict[str, dict] = {}
    for cat in categories:
        cat_df = df[df["Category"] == cat]
        k_cat  = cat_df["Predicted_Label"].nunique()
        pv     = compute_bp_per_vignette(cat_df, k_categories=k_cat)
        bp_by_cat_pv[cat]      = pv
        bp_by_cat_summary[cat] = summarise_bp(pv, cat)

    # ------------------------------------------------------------------
    # 3. Merge into a single summary table
    # ------------------------------------------------------------------
    summary_records = []
    for subset_name in ["Overall"] + categories:
        ka  = ka_overall if subset_name == "Overall" else ka_by_cat[subset_name]
        bp  = bp_overall_summary if subset_name == "Overall" else bp_by_cat_summary[subset_name]
        summary_records.append({
            "Subset":                    subset_name,
            "n_vignettes":               ka["n_vignettes"],
            "mean_raters_per_vignette":  ka["mean_raters_per_vignette"],
            # Krippendorff
            "kripp_alpha":               ka["alpha"],
            "kripp_ci_lower":            ka["ci_lower"],
            "kripp_ci_upper":            ka["ci_upper"],
            "kripp_ci":                  ka["ci"],
            "kripp_n_bootstrap":         ka["n_bootstrap"],
            # Brennan-Prediger
            "bp_mean":                   bp["bp_mean"],
            "bp_std":                    bp["bp_std"],
            "bp_median":                 bp["bp_median"],
            "bp_ci_lower":               bp["bp_ci_lower"],
            "bp_ci_upper":               bp["bp_ci_upper"],
            "bp_ci":                     bp["bp_ci"],
        })

    summary_df = pd.DataFrame(summary_records)

    # ------------------------------------------------------------------
    # 4. Print compact summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INTER-RATER AGREEMENT SUMMARY")
    print("=" * 80)

    print(f"\n{'Subset':<10} {'Krippendorff α':>16}  {'95% CI (bootstrap)':<24}"
          f"{'BP κ mean ± SD':>18}  {'95% CI (t)'}")
    print("-" * 80)
    for rec in summary_records:
        print(
            f"{rec['Subset']:<10} "
            f"{rec['kripp_alpha']:>16.4f}  "
            f"{rec['kripp_ci']:<24}"
            f"{rec['bp_mean']:>8.4f} ± {rec['bp_std']:<8.4f}  "
            f"{rec['bp_ci']}"
        )
    print("-" * 80)
    print("  Krippendorff α: single value over all vignettes jointly (nominal).")
    print("  Brennan-Prediger κ: mean ± SD of per-vignette values [95% t CI].")
    print("  BP uses uniform chance baseline (1/k), correcting for prevalence skew.")
    print(f"  k_overall = {k_overall} labels; per-category k computed separately.\n")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    sheets: dict[str, pd.DataFrame] = {
        "Summary":           summary_df,
        "BP_Per_Vignette":   bp_overall_pv.reset_index(),
    }
    for cat in categories:
        sheets[f"BP_Per_Vignette_{cat}"] = bp_by_cat_pv[cat].reset_index()

    save_excel(sheets, OUTPUT_XLSX, index=False)
    append_to_metrics(summary_df, METRICS_XLSX)

    print(f"All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()