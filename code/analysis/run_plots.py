"""
run_plots.py
============
Generate publication-quality plots from saved classification metrics.

Loads from:
    results_folder/classification_metrics_output/<language>/classification_metrics.xlsx

Produces (saved to results_folder/_Figures/):
    1. English_accuracy_kappa.pdf
       — Grouped bar chart: accuracy + kappa side-by-side per model,
         both with 95% CIs. Clinicians included in accuracy bar only
         (hatched). Models follow canonical MODEL_ORDER.

    2. English_sens_spec_f1.pdf
       — Grouped bar chart: weighted sensitivity, weighted specificity,
         weighted F1 per model. No CIs (no natural sampling distribution
         for these aggregate metrics).

    3. Multilingual_accuracy.pdf
       — Grouped bar chart: languages on x-axis, hue = model,
         accuracy + 95% CIs. English panel included.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import json

from icd11_utils import (
    LANGUAGES,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    sort_models,
)

sns.set_context("paper", font_scale=1.4)

# ---------------------------------------------------------------------------
# Configuration — update paths here
# ---------------------------------------------------------------------------
def load_config(file):
    with open(file) as f:
        config_dict = json.load(f)
    return config_dict

config_dict = load_config(file=Path(__file__).parents[1].joinpath("config_paths.json"))["hpc"]
BASE_PATH = Path(config_dict['base_path'])
RESULTS_FOLDER = BASE_PATH / "results_resubmission"
METRICS_FOLDER = RESULTS_FOLDER / "_results"
FIGURES_FOLDER = RESULTS_FOLDER / "_figures" / "weighted_vignettes"
FIGURES_FOLDER.mkdir(exist_ok=True)

FIGURES_FOLDER_ABLATION = RESULTS_FOLDER / "ablation" / "_figures" / "weighted_vignettes"
FIGURES_FOLDER_ABLATION.mkdir(exist_ok=True)

CLINICIAN_CSV = RESULTS_FOLDER / "_results" / "clinicians" / "clinician_classification_metrics.xlsx"

LANGUAGE_DISPLAY = {
    "english":  "English",
    "spanish":  "Spanish",
    "chinese":  "Chinese",
    "french":   "French",
    "japanese": "Japanese",
    "russian":  "Russian",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_overall_sheet(language: str, path: str=None) -> pd.DataFrame | None:
    """Load the Overall sheet from the classification_metrics Excel for one language."""
    if path:
        df = pd.read_excel(path, sheet_name="Overall", index_col="llm")
    else:
        xl_path = METRICS_FOLDER / language / "classification_metrics.xlsx"
        if not xl_path.exists():
            print(f"  WARNING: {xl_path} not found, skipping.")
            return None
        df = pd.read_excel(xl_path, sheet_name="Overall", index_col="llm")
    return df


def load_clinicians() -> dict:
    """Load clinician metrics from the Overall sheet."""
    df = pd.read_excel(CLINICIAN_CSV, sheet_name="Overall")
    row = df.iloc[0]

    def _get(col):
        return round(row[col], 4) if col in df.columns else None

    return {
        "accuracy":             _get("vw_accuracy"),
        "CI_lower":             _get("vw_CI_lower"),
        "CI_upper":             _get("vw_CI_upper"),
        "kappa":                None,
        "kappa_CI_lower":       None,
        "kappa_CI_upper":       None,
        "weighted_f1":          _get("weighted_f1"),
        "weighted_sensitivity": _get("weighted_sensitivity"),
        "weighted_specificity": _get("weighted_specificity"),
        "weighted_precision":   _get("weighted_precision"),
        "macro_f1":             _get("macro_f1"),
        "macro_sensitivity":    _get("macro_sensitivity"),
        "macro_specificity":    _get("macro_specificity"),
        "macro_precision":      _get("macro_precision"),
    }


def _add_ci_errorbars(ax, x_positions, means, lowers, uppers, **kwargs):
    """Add asymmetric CI error bars to an axes."""
    lower_err = np.array(means) - np.array(lowers)
    upper_err = np.array(uppers) - np.array(means)
    ax.errorbar(
        x_positions, means,
        yerr=[lower_err, upper_err],
        fmt="none",
        capsize=3,
        ecolor="black",
        elinewidth=0.9,
        zorder=5,
        **kwargs,
    )


def _apply_hatch(ax, bar_indices: list[int], hatch: str = "/"):
    """Apply hatching to specific bar patches by index."""
    patches = [p for p in ax.patches if hasattr(p, "get_height")]
    for i in bar_indices:
        if i < len(patches):
            patches[i].set_hatch(hatch)
            patches[i].set_edgecolor("black")


# ---------------------------------------------------------------------------
# Plot 1: English accuracy + kappa (side-by-side grouped bars)
# ---------------------------------------------------------------------------

def plot_english_accuracy_kappa(df_english: pd.DataFrame, clinician: dict, path:str=None):
    """
    Two side-by-side subplots sharing a y-axis:
      Left:  Accuracy + 95% CI per model (+ Clinicians, hatched)
      Right: Cohen's kappa + 95% CI per model (no Clinicians)
    Uses 3 colours: open-source, closed-source, clinicians.
    """
    COLOR_OPEN   = "#D0CAF0"
    COLOR_CLOSED = "#9287C1"
    COLOR_CLI    = "#B2CDE7"  # "#70B69E"
    CLOSED_SOURCE_MODELS = { "gemini_25", "gpt_51", "claude_opus46"}

    models_llm = sort_models(df_english.index.tolist())
    models_acc = models_llm + ["Clinicians"]
    models_kap = models_llm

    def _color(m):
        if m == "Clinicians":
            return COLOR_CLI
        return COLOR_CLOSED if m in CLOSED_SOURCE_MODELS else COLOR_OPEN

    def _vals(models, col, multiplier=100, clinician_val=None):
        out = []
        for m in models:
            if m == "Clinicians":
                out.append(clinician_val * multiplier)
            else:
                out.append(df_english.loc[m, col] * multiplier)
        return out

    acc_vals  = _vals(models_acc, "accuracy",  clinician_val=clinician["accuracy"])
    acc_lower = _vals(models_acc, "CI_lower",  clinician_val=clinician["CI_lower"])
    acc_upper = _vals(models_acc, "CI_upper",  clinician_val=clinician["CI_upper"])

    kap_vals  = _vals(models_kap, "kappa")
    kap_lower = _vals(models_kap, "kappa_CI_lower")
    kap_upper = _vals(models_kap, "kappa_CI_upper")

    n_acc = len(models_acc)
    n_kap = len(models_kap)
    width = 0.62

    fig, (ax_acc, ax_kap) = plt.subplots(
        1, 2,
        figsize=(max(12, n_acc * 0.95 + 3), 5),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    def _draw_bars(ax, models, vals, lowers, uppers, hatch_clinicians=False):
        x = np.arange(len(models))
        for i, m in enumerate(models):
            bar = ax.bar(
                x[i], vals[i], width,
                color=_color(m), edgecolor="black", linewidth=0.8, zorder=3,
            )
            if hatch_clinicians and m == "Clinicians":
                bar[0].set_hatch("///")
                bar[0].set_edgecolor("black")
                bar[0].set_linewidth(0.8)

        lower_err = np.array(vals) - np.array(lowers)
        upper_err = np.array(uppers) - np.array(vals)
        ax.errorbar(
            x, vals,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=3,
            ecolor="#333333", elinewidth=1.0, capthick=1.0,
            zorder=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(m, m) for m in models],
            rotation=45, ha="right", fontsize=10,
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _draw_bars(ax_acc, models_acc, acc_vals, acc_lower, acc_upper, hatch_clinicians=True)
    _draw_bars(ax_kap, models_kap, kap_vals, kap_lower, kap_upper)

    ax_acc.set_ylabel("Score (%)", fontsize=12)
    ax_acc.set_ylim(0, 110)
    ax_acc.set_title("Accuracy", fontsize=14, fontweight="bold", pad=6)
    ax_kap.set_title("Cohen's κ", fontsize=14, fontweight="bold", pad=6)

    ax_kap.spines["left"].set_visible(False)
    ax_kap.tick_params(left=False)

    handles = [
        mpatches.Patch(facecolor=COLOR_OPEN,   label="Open-source", edgecolor="black"),
        mpatches.Patch(facecolor=COLOR_CLOSED, label="Closed-source", edgecolor="black"),
        mpatches.Patch(facecolor=COLOR_CLI,    label="Clinicians", hatch="//////", edgecolor="#333333", linewidth=1.0),
    ]
    fig.legend(
        handles=handles,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout()
    if path:
        out = path
    else:
        out = FIGURES_FOLDER / "English_accuracy_kappa.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

def plot_english_accuracy_kappa2(df_english: pd.DataFrame, clinician: dict, path:str=None):
    """
    Two side-by-side subplots sharing a y-axis:
      Left:  Accuracy + 95% CI per model (+ Clinicians, hatched)
      Right: Cohen's kappa + 95% CI per model (no Clinicians)
    Uses 3 colours: open-source, closed-source, clinicians.
    """
    COLOR_OPEN   = "#D0CAF0"
    COLOR_CLOSED = "#9287C1"
    COLOR_CLI    = "#B2CDE7"  # "#70B69E"
    CLOSED_SOURCE_MODELS = { "gemini_25", "gpt_51", "claude_opus46"}

    models_llm = sort_models(df_english.index.tolist())
    models_acc = models_llm + ["Clinicians"]
    models_kap = models_llm

    def _color(m):
        if m == "Clinicians":
            return COLOR_CLI
        return COLOR_CLOSED if m in CLOSED_SOURCE_MODELS else COLOR_OPEN

    def _vals(models, col, multiplier=100, clinician_val=None):
        out = []
        for m in models:
            if m == "Clinicians":
                out.append(clinician_val * multiplier)
            else:
                out.append(df_english.loc[m, col] * multiplier)
        return out

    acc_vals  = _vals(models_acc, "accuracy",  clinician_val=clinician["accuracy"])
    acc_lower = _vals(models_acc, "CI_lower",  clinician_val=clinician["CI_lower"])
    acc_upper = _vals(models_acc, "CI_upper",  clinician_val=clinician["CI_upper"])

    kap_vals  = _vals(models_kap, "kappa") 
    kap_lower = _vals(models_kap, "kappa_CI_lower") 
    kap_upper = _vals(models_kap, "kappa_CI_upper") 

    n_acc = len(models_acc)
    n_kap = len(models_kap)
    width = 0.62

    fig, (ax_acc, ax_kap) = plt.subplots(
        1, 2,
        figsize=(max(12, n_acc * 0.95 + 3), 5),
        # sharey=True,
        # gridspec_kw={"wspace": 0.06},
    )

    def _draw_bars(ax, models, vals, lowers, uppers, hatch_clinicians=False):
        x = np.arange(len(models))
        for i, m in enumerate(models):
            bar = ax.bar(
                x[i], vals[i], width,
                color=_color(m), edgecolor="black", linewidth=0.8, zorder=3,
            )
            if hatch_clinicians and m == "Clinicians":
                bar[0].set_hatch("///")
                bar[0].set_edgecolor("black")
                bar[0].set_linewidth(0.8)

        lower_err = np.array(vals) - np.array(lowers)
        upper_err = np.array(uppers) - np.array(vals)
        ax.errorbar(
            x, vals,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=3,
            ecolor="#333333", elinewidth=1.0, capthick=1.0,
            zorder=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(m, m) for m in models],
            rotation=45, ha="right", fontsize=10,
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _draw_bars(ax_acc, models_acc, acc_vals, acc_lower, acc_upper, hatch_clinicians=True)
    _draw_bars(ax_kap, models_kap, np.array(kap_vals) / 100, np.array(kap_lower) / 100, np.array(kap_upper) / 100)

    ax_acc.set_ylabel("Accuracy (%)", fontsize=12)
    ax_acc.set_title("Mean Accuracy", fontsize=14, fontweight='bold')
    ax_acc.set_ylim(0, 110)

    ax_kap.set_ylabel("Cohen's κ", fontsize=12)
    ax_kap.set_title("LLM-Clinician Agreement", fontsize=14, fontweight='bold')
    ax_kap.set_ylim(0, 1.1)


    handles = [
        mpatches.Patch(facecolor=COLOR_OPEN,   label="Open-source", edgecolor="black"),
        mpatches.Patch(facecolor=COLOR_CLOSED, label="Closed-source", edgecolor="black"),
        mpatches.Patch(facecolor=COLOR_CLI,    label="Clinicians", hatch="//////", edgecolor="#333333", linewidth=1.0),
    ]
    fig.legend(
        handles=handles,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout()
    if path:
        out = path
    else:
        out = FIGURES_FOLDER / "English_accuracy_kappa2.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

def plot_english_accuracy_kappa_refline(df_english: pd.DataFrame, clinician: dict, path: str = None):
    """
    Two side-by-side subplots sharing a y-axis:
      Left:  Accuracy + 95% CI per model, with clinician accuracy shown as
             a horizontal reference line + grey CI band (no clinician bar).
      Right: Cohen's kappa + 95% CI per model (LLMs only).
    Uses 2 colours: open-source, closed-source.
    """
    COLOR_OPEN   = "#D0CAF0"
    COLOR_CLOSED = "#9287C1"
    CLOSED_SOURCE_MODELS = {"gemini_25", "gpt_51", "claude_opus46"}

    models_llm = sort_models(df_english.index.tolist())

    def _color(m):
        return COLOR_CLOSED if m in CLOSED_SOURCE_MODELS else COLOR_OPEN

    def _vals(models, col, multiplier=100):
        return [df_english.loc[m, col] * multiplier for m in models]

    acc_vals  = _vals(models_llm, "accuracy")
    acc_lower = _vals(models_llm, "CI_lower")
    acc_upper = _vals(models_llm, "CI_upper")

    kap_vals  = _vals(models_llm, "kappa")
    kap_lower = _vals(models_llm, "kappa_CI_lower")
    kap_upper = _vals(models_llm, "kappa_CI_upper")

    # Clinician reference values (in %)
    cli_mean = clinician["accuracy"] * 100
    cli_lo   = clinician["CI_lower"] * 100
    cli_hi   = clinician["CI_upper"] * 100

    n = len(models_llm)
    width = 0.62

    fig, (ax_acc, ax_kap) = plt.subplots(
        1, 2,
        figsize=(max(12, n * 0.9 + 3), 4.5),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    def _draw_bars(ax, models, vals, lowers, uppers):
        x = np.arange(len(models))
        for i, m in enumerate(models):
            ax.bar(
                x[i], vals[i], width,
                color=_color(m), edgecolor="black", linewidth=0.8, zorder=3,
            )

        lower_err = np.array(vals) - np.array(lowers)
        upper_err = np.array(uppers) - np.array(vals)
        ax.errorbar(
            x, vals,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=3,
            ecolor="#333333", elinewidth=1.0, capthick=1.0,
            zorder=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(m, m) for m in models],
            rotation=45, ha="right", fontsize=10,
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Accuracy panel ---
    # Clinician reference band + line (drawn first so bars sit on top)
    ax_acc.axhspan(cli_lo, cli_hi, color="#D9D9D9", alpha=0.5, zorder=1)
    ax_acc.axhline(cli_mean, color="#666666", linewidth=1.2, linestyle="--", zorder=2)

    _draw_bars(ax_acc, models_llm, acc_vals, acc_lower, acc_upper)

    # --- Kappa panel ---
    _draw_bars(ax_kap, models_llm, kap_vals, kap_lower, kap_upper)

    ax_acc.set_ylabel("Score (%)", fontsize=12)
    ax_acc.set_ylim(0, 100)
    ax_acc.set_title("Accuracy", fontsize=14, fontweight="bold", pad=6)
    ax_kap.set_title("Cohen's κ", fontsize=14, fontweight="bold", pad=6)

    ax_kap.spines["left"].set_visible(False)
    ax_kap.tick_params(left=False)

    # Legend
    handles = [
        mpatches.Patch(facecolor=COLOR_OPEN, label="Open-source", edgecolor="black"),
        mpatches.Patch(facecolor=COLOR_CLOSED, label="Closed-source", edgecolor="black"),
        plt.Line2D([0], [0], color="#666666", linewidth=1.2, linestyle="--", label="Clinician Accuracy"),
        mpatches.Patch(facecolor="#D9D9D9", alpha=0.5, edgecolor="#BBBBBB", label="Clinician 95% CI"),
    ]
    fig.legend(
        handles=handles,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout()
    if path:
        out = path
    else:
        out = FIGURES_FOLDER / "English_accuracy_kappa_refline.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 1b: English accuracy + kappa — 2-colour version (single panel)
# ---------------------------------------------------------------------------

def plot_english_accuracy_kappa_2color(df_english: pd.DataFrame, clinician: dict, path:str=None):
    """
    Single grouped bar chart: accuracy (dark) + kappa (light) side-by-side
    per model, using only 2 colours to distinguish the two metrics.
    Clinicians included in accuracy bar only (hatched).
    """
    COLOR_ACC = "#5068A8"   # slightly lighter dark blue for accuracy
    COLOR_CLI = "#B2CDE7"   # muted green for clinicians
    COLOR_KAP = "#C5D4F0"   # lighter blue for kappa

    models_llm = sort_models(df_english.index.tolist())
    models_all = models_llm + ["Clinicians"]

    acc_vals  = [df_english.loc[m, "accuracy"]      * 100 if m != "Clinicians" else clinician["accuracy"]  * 100 for m in models_all]
    acc_lower = [df_english.loc[m, "CI_lower"]      * 100 if m != "Clinicians" else clinician["CI_lower"]  * 100 for m in models_all]
    acc_upper = [df_english.loc[m, "CI_upper"]      * 100 if m != "Clinicians" else clinician["CI_upper"]  * 100 for m in models_all]
    kap_vals  = [df_english.loc[m, "kappa"]         * 100 if m != "Clinicians" else None for m in models_all]
    kap_lower = [df_english.loc[m, "kappa_CI_lower"]* 100 if m != "Clinicians" else None for m in models_all]
    kap_upper = [df_english.loc[m, "kappa_CI_upper"]* 100 if m != "Clinicians" else None for m in models_all]

    n = len(models_all)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, n * 1.0), 5))

    for i, m in enumerate(models_all):
        # Accuracy bar
        bar_color = COLOR_CLI if m == "Clinicians" else COLOR_ACC
        acc_bar = ax.bar(
            x[i] - width / 2, acc_vals[i], width,
            color=bar_color, edgecolor="black", linewidth=0.3, zorder=3,
        )
        if m == "Clinicians":
            acc_bar[0].set_hatch("///")
            acc_bar[0].set_edgecolor("black")
            acc_bar[0].set_linewidth(0.8)

        # Kappa bar (skip Clinicians)
        if kap_vals[i] is not None:
            ax.bar(
                x[i] + width / 2, kap_vals[i], width,
                color=COLOR_KAP, edgecolor="black", linewidth=0.3, zorder=3,
            )

    # CI error bars — accuracy
    _add_ci_errorbars(ax, x - width / 2, acc_vals, acc_lower, acc_upper)

    # CI error bars — kappa
    kap_x     = [x[i] + width / 2 for i in range(n) if kap_vals[i] is not None]
    kap_means = [v for v in kap_vals  if v is not None]
    kap_lows  = [v for v in kap_lower if v is not None]
    kap_highs = [v for v in kap_upper if v is not None]
    _add_ci_errorbars(ax, kap_x, kap_means, kap_lows, kap_highs)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_all],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    acc_patch = mpatches.Patch(color=COLOR_ACC, label="Accuracy (LLMs)", edgecolor="black")
    kap_patch = mpatches.Patch(color=COLOR_KAP, label="Cohen's κ (LLMs)", edgecolor="black")
    cli_patch = mpatches.Patch(color=COLOR_CLI, label="Accuracy (Clinicians)", hatch="///", edgecolor="black")
    ax.legend(
        handles=[acc_patch, kap_patch, cli_patch],
        title="Metric", fontsize=9, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0, framealpha=0.9, edgecolor="black",
    )

    plt.tight_layout()
    if path:
        out = path
    else:
        out = FIGURES_FOLDER / "English_accuracy_kappa_2color.pdf"
    
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 2: English weighted sensitivity / specificity / F1
# ---------------------------------------------------------------------------

def plot_english_sens_spec_f1_weighted(df_english: pd.DataFrame, clinician: dict):
    models = sort_models(df_english.index.tolist())
    models_all = models + ["Clinicians"]
    metrics = {
        "F1":          "weighted_f1",
        "Sensitivity": "weighted_sensitivity",
        "Specificity": "weighted_specificity",
    }
    metric_colors = ["#5068A8", "#89ABEB", "#C5D4F0"]
    COLOR_CLI = "#B2CDE7"

    n = len(models_all)
    n_metrics = len(metrics)
    x = np.arange(n)
    width = 0.25
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 6))

    for j, (label, col) in enumerate(metrics.items()):
        for i, m in enumerate(models_all):
            if m == "Clinicians":
                val = clinician.get(col)
                if val is None:
                    continue
                val *= 100
                bar = ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    label=f"Clinicians ({label})" if i == len(models_all) - 1 else "_nolegend_",
                    zorder=3,
                )
                bar[0].set_hatch("///")
            else:
                val = df_english.loc[m, col] * 100
                ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    label=label if i == 0 else "_nolegend_",
                    zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_all],
        rotation=45, ha="right",
    )
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    metric_handles = [
        mpatches.Patch(color=c, label=l, edgecolor="black")
        for (l, _), c in zip(metrics.items(), metric_colors)
    ]
    cli_handle = mpatches.Patch(facecolor="#FFFFFF", label="Clinicians", hatch="///", edgecolor="black")
    ax.legend(
        handles=metric_handles + [cli_handle],
        title="Metric", fontsize=12, title_fontsize=12,
        loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=4,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )
    plt.tight_layout()
    out = FIGURES_FOLDER / "English_sens_spec_f1_weighted.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_english_sens_spec_f1_macro(df_english: pd.DataFrame, clinician: dict):
    models = sort_models(df_english.index.tolist())
    models_all = models + ["Clinicians"]
    metrics = {
        "F1":          "macro_f1",
        "Sensitivity": "macro_sensitivity",
        "Specificity": "macro_specificity",
    }
    metric_colors = ["#5068A8", "#89ABEB", "#C5D4F0"]
    COLOR_CLI = "#B2CDE7"

    n = len(models_all)
    n_metrics = len(metrics)
    x = np.arange(n)
    width = 0.25
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 6))

    for j, (label, col) in enumerate(metrics.items()):
        for i, m in enumerate(models_all):
            if m == "Clinicians":
                val = clinician.get(col)
                if val is None:
                    continue
                val *= 100
                bar = ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    zorder=3,
                )
                bar[0].set_hatch("///")
            else:
                val = df_english.loc[m, col] * 100
                ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    label=label if i == 0 else "_nolegend_",
                    zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_all],
        rotation=45, ha="right", fontsize=12,
    )
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    metric_handles = [
        mpatches.Patch(color=c, label=l, edgecolor="black")
        for (l, _), c in zip(metrics.items(), metric_colors)
    ]
    cli_handle = mpatches.Patch(facecolor="#FFFFFF", label="Clinicians", hatch="///", edgecolor="black")
    ax.legend(
        handles=metric_handles + [cli_handle],
        title="Metric", fontsize=12, title_fontsize=12,
        loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=4,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )
    plt.tight_layout()
    out = FIGURES_FOLDER / "English_sens_spec_f1_macro.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_english_sens_prec_f1_weighted(df_english: pd.DataFrame, clinician: dict):
    models = sort_models(df_english.index.tolist())
    models_all = models + ["Clinicians"]
    metrics = {
        "F1":        "weighted_f1",
        "Sensitivity": "weighted_sensitivity",
        "Precision": "weighted_precision",
    }
    metric_colors = ["#5068A8", "#89ABEB", "#C5D4F0"]
    COLOR_CLI = "#B2CDE7"

    n = len(models_all)
    n_metrics = len(metrics)
    x = np.arange(n)
    width = 0.25
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 6))

    for j, (label, col) in enumerate(metrics.items()):
        for i, m in enumerate(models_all):
            if m == "Clinicians":
                val = clinician.get(col)
                if val is None:
                    continue
                val *= 100
                bar = ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    zorder=3,
                )
                bar[0].set_hatch("///")
            else:
                val = df_english.loc[m, col] * 100
                ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    label=label if i == 0 else "_nolegend_",
                    zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_all],
        rotation=45, ha="right", fontsize=12,
    )
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    metric_handles = [
        mpatches.Patch(color=c, label=l, edgecolor="black")
        for (l, _), c in zip(metrics.items(), metric_colors)
    ]
    cli_handle = mpatches.Patch(facecolor="#FFFFFF", label="Clinicians", hatch="///", edgecolor="black")
    ax.legend(
        handles=metric_handles + [cli_handle],
        title="Metric", fontsize=12, title_fontsize=12,
        loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=4,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )
    plt.tight_layout()
    out = FIGURES_FOLDER / "English_sens_prec_f1_weighted.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

def plot_english_sens_spec_prec_f1_weighted(df_english: pd.DataFrame, clinician: dict):
    models = sort_models(df_english.index.tolist())
    models_all = models + ["Clinicians"]
    metrics = {
        "F1":        "weighted_f1",
        "Sensitivity": "weighted_sensitivity",
        "Precision": "weighted_precision",
        "Specificity": "weighted_specificity"
    }
    metric_colors = ["#5068A8", "#5C8CE5","#91ABDC", "#C5D4F0"]
    COLOR_CLI = "#B2CDE7"

    n = len(models_all)
    n_metrics = len(metrics)
    x = np.arange(n)
    width = 0.2
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    for j, (label, col) in enumerate(metrics.items()):
        for i, m in enumerate(models_all):
            if m == "Clinicians":
                val = clinician.get(col)
                if val is None:
                    continue
                val *= 100
                bar = ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    zorder=3,
                )
                bar[0].set_hatch("///")
            else:
                val = df_english.loc[m, col] * 100
                ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    label=label if i == 0 else "_nolegend_",
                    zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_all],
        rotation=45, ha="right", fontsize=12,
    )
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    metric_handles = [
        mpatches.Patch(color=c, label=l, edgecolor="black")
        for (l, _), c in zip(metrics.items(), metric_colors)
    ]

    ax.legend(
        handles=metric_handles,
        title="Metric", fontsize=12, title_fontsize=12,
        loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=4,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )
    plt.tight_layout()
    out = FIGURES_FOLDER / "English_sens_spec_prec_f1_weighted.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_english_sens_prec_f1_macro(df_english: pd.DataFrame, clinician: dict):
    models = sort_models(df_english.index.tolist())
    models_all = models + ["Clinicians"]
    metrics = {
        "F1":          "macro_f1",
        "Sensitivity": "macro_sensitivity",
        "Precision":   "macro_precision",
    }
    metric_colors = ["#5068A8", "#89ABEB", "#C5D4F0"]
    COLOR_CLI = "#B2CDE7"

    n = len(models_all)
    n_metrics = len(metrics)
    x = np.arange(n)
    width = 0.25
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 6))

    for j, (label, col) in enumerate(metrics.items()):
        for i, m in enumerate(models_all):
            if m == "Clinicians":
                val = clinician.get(col)
                if val is None:
                    continue
                val *= 100
                bar = ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    zorder=3,
                )
                bar[0].set_hatch("///")
            else:
                val = df_english.loc[m, col] * 100
                ax.bar(
                    x[i] + offsets[j], val, width,
                    color=metric_colors[j], edgecolor="black", linewidth=0.7,
                    label=label if i == 0 else "_nolegend_",
                    zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_all],
        rotation=45, ha="right", fontsize=12,
    )
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    metric_handles = [
        mpatches.Patch(color=c, label=l, edgecolor="black")
        for (l, _), c in zip(metrics.items(), metric_colors)
    ]
    cli_handle = mpatches.Patch(facecolor="#FFFFFF", label="Clinicians", hatch="///", edgecolor="black")
    ax.legend(
        handles=metric_handles + [cli_handle],
        title="Metric", fontsize=12, title_fontsize=12,
        loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=4,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )
    plt.tight_layout()
    out = FIGURES_FOLDER / "English_sens_prec_f1_macro.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 4: English accuracy per category (Anxiety, Mood, Stress)
# ---------------------------------------------------------------------------

def plot_english_per_category(language: str = "english"):
    """
    Grouped bar chart: models on x-axis, hue = category (3 colours),
    accuracy + 95% CI error bars. Clinicians added as a separate bar group
    using per-category accuracy from the clinician CSV.
    """
    xl_path = METRICS_FOLDER / language / "classification_metrics.xlsx"
    if not xl_path.exists():
        print(f"  WARNING: {xl_path} not found, skipping.")
        return

    CATEGORIES = ["Anxiety", "Mood", "Stress"]
    CAT_COLORS = ["#7D4FBE","#A084C9", "#E9E2F3"]  # blue, green, orange

    # --- Load LLM per-category sheets ---
    cat_data = {}
    for cat in CATEGORIES:
        try:
            cat_data[cat] = pd.read_excel(xl_path, sheet_name=cat, index_col="llm")
        except Exception:
            print(f"  WARNING: sheet '{cat}' not found, skipping.")
    if not cat_data:
        return

    # --- Load clinician per-category data ---
    cli_cat = {}
    if CLINICIAN_CSV.exists():
        for cat in CATEGORIES:
            df = pd.read_excel(CLINICIAN_CSV, sheet_name=cat)
            cli_cat[cat] = {
                "accuracy":  round(df.loc[0, 'accuracy']  * 100, 2),
                "CI_lower":  round(df.loc[0, 'CI_lower']  * 100, 2),
                "CI_upper":  round(df.loc[0, 'CI_upper']  * 100, 2),
            }

    # --- Build model list: LLMs + Clinicians ---
    all_models = set()
    for df in cat_data.values():
        all_models.update(df.index.tolist())
    models_llm = sort_models(list(all_models))
    models_all = models_llm + (["Clinicians"] if cli_cat else [])

    n_models = len(models_all)
    n_cat = len(CATEGORIES)
    bar_group_width = 0.7
    width = bar_group_width / n_cat
    x = np.arange(n_models)
    offsets = np.linspace(-(n_cat - 1) / 2, (n_cat - 1) / 2, n_cat) * width

    fig_width = n_models * 1
    fig, ax = plt.subplots(figsize=(fig_width, 3))

    for j, (cat, color) in enumerate(zip(CATEGORIES, CAT_COLORS)):
        for i, m in enumerate(models_all):
            if m == "Clinicians":
                if cat not in cli_cat:
                    continue
                val  = cli_cat[cat]["accuracy"]
                low  = cli_cat[cat]["CI_lower"]
                high = cli_cat[cat]["CI_upper"]
            else:
                if cat not in cat_data or m not in cat_data[cat].index:
                    continue
                val  = cat_data[cat].loc[m, "accuracy"] * 100
                low  = cat_data[cat].loc[m, "CI_lower"] * 100
                high = cat_data[cat].loc[m, "CI_upper"] * 100

            bar = ax.bar(
                x[i] + offsets[j], val, width,
                color=color, edgecolor="black", linewidth=0.3,
                label=cat if i == 0 else "_nolegend_", zorder=3,
            )
            # Hatch clinician bars
            if m == "Clinicians":
                bar[0].set_hatch("///")
                bar[0].set_edgecolor("black")
                bar[0].set_linewidth(0.6)

            ax.errorbar(
                x[i] + offsets[j], val,
                yerr=[[val - low], [high - val]],
                fmt="none", capsize=2,
                ecolor="#333333", elinewidth=0.7, capthick=0.7,
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_all],
        rotation=45, ha="right", fontsize=12,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Category", fontsize=9, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout()
    out = FIGURES_FOLDER / "English_accuracy_per_category.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

def plot_english_per_category_subplots(
    metrics_folder,
    clinician_csv,
    figures_folder,
    language: str = "english",
):
    """
    One subplot per category (Anxiety, Mood, Stress): models on x-axis,
    accuracy bars in uniform colour with 95% CI error bars.
    Clinicians included with hatched bars.

    Parameters
    ----------
    metrics_folder : Path
        Folder containing <language>/classification_metrics.xlsx.
    clinician_csv : Path
        Path to clinician detailed CSV.
    figures_folder : Path
        Output folder for the PDF.
    language : str
        Language subfolder to load (default "english").
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from icd11_utils import MODEL_DISPLAY_NAMES, sort_models

    xl_path = metrics_folder / language / "classification_metrics.xlsx"
    if not xl_path.exists():
        print(f"  WARNING: {xl_path} not found, skipping.")
        return

    COLOR_OPEN   = "#D0CAF0"   # lighter purple — open-source
    COLOR_CLOSED = "#9287C1"   # darker purple — closed-source
    COLOR_CLI    = "#B2CDE7"   # gold — clinicians

    CLOSED_SOURCE_MODELS = { "gemini_25", "gpt_51", "claude_opus46"}

    CATEGORIES = ["Anxiety", "Mood", "Stress"]

    # --- Load LLM per-category sheets ---
    cat_data = {}
    for cat in CATEGORIES:
        try:
            cat_data[cat] = pd.read_excel(xl_path, sheet_name=cat, index_col="llm")
        except Exception:
            print(f"  WARNING: sheet '{cat}' not found, skipping.")
    if not cat_data:
        return

    # --- Load clinician per-category data ---
    cli_cat = {}
    if clinician_csv.exists():
        for cat in CATEGORIES:
            df = pd.read_excel(CLINICIAN_CSV, sheet_name=cat)
            cli_cat[cat] = {
                "accuracy":  round(df.loc[0, 'vw_accuracy'] * 100, 2),
                "CI_lower":  round(df.loc[0, 'vw_CI_lower']  * 100, 2),
                "CI_upper":  round(df.loc[0, 'vw_CI_upper']  * 100, 2),
            }

    # --- Build model list ---
    all_models = set()
    for df in cat_data.values():
        all_models.update(df.index.tolist())
    models_llm = sort_models(list(all_models))
    models_all = models_llm + (["Clinicians"] if cli_cat else [])

    n_models = len(models_all)
    ncols = len(CATEGORIES)
    x = np.arange(n_models)
    width = 0.65

    fig, axes = plt.subplots(
        1, ncols,
        figsize=(ncols * 4, 4),
        sharey=True,
    )

    for j, cat in enumerate(CATEGORIES):
        ax = axes[j]

        vals, ci_los, ci_his = [], [], []
        is_clinician = []

        for m in models_all:
            if m == "Clinicians":
                if cat in cli_cat:
                    vals.append(cli_cat[cat]["accuracy"])
                    ci_los.append(cli_cat[cat]["CI_lower"])
                    ci_his.append(cli_cat[cat]["CI_upper"])
                else:
                    vals.append(0)
                    ci_los.append(0)
                    ci_his.append(0)
                is_clinician.append(True)
            else:
                if cat in cat_data and m in cat_data[cat].index:
                    vals.append(cat_data[cat].loc[m, "accuracy"] * 100)
                    ci_los.append(cat_data[cat].loc[m, "CI_lower"] * 100)
                    ci_his.append(cat_data[cat].loc[m, "CI_upper"] * 100)
                else:
                    vals.append(0)
                    ci_los.append(0)
                    ci_his.append(0)
                is_clinician.append(False)

        for i in range(n_models):
            m = models_all[i]
            if is_clinician[i]:
                color = COLOR_CLI
            elif m in CLOSED_SOURCE_MODELS:
                color = COLOR_CLOSED
            else:
                color = COLOR_OPEN

            bar = ax.bar(
                x[i], vals[i], width,
                color=color, edgecolor="black", linewidth=0.8, zorder=3,
            )
            if is_clinician[i]:
                bar[0].set_hatch("///")
                bar[0].set_edgecolor("black")
                bar[0].set_linewidth(0.8)

        lower_err = np.array(vals) - np.array(ci_los)
        upper_err = np.array(ci_his) - np.array(vals)
        ax.errorbar(
            x, vals,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=2.5,
            ecolor="#333333", elinewidth=1.0, capthick=1.0,
            zorder=5,
        )

        ax.set_title(cat, fontsize=11, fontweight="bold", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(m, m) for m in models_all],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_ylim(0, 105)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if j == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10)

    # Shared legend
    handles = [
        mpatches.Patch(color=COLOR_OPEN, label="Open-Source", edgecolor="black"),
        mpatches.Patch(color=COLOR_CLOSED, label="Closed-Source", edgecolor="black"),
        mpatches.Patch(facecolor=COLOR_CLI,    label="Clinicians", hatch="//////", edgecolor="#333333", linewidth=1.0)
    ]
    fig.legend(
        handles=handles,
        fontsize=9, title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out = figures_folder / "English_accuracy_per_category_subplots.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 3: Multilingual accuracy (languages on x-axis, hue = model)
# ---------------------------------------------------------------------------

def plot_multilingual_accuracy(lang_data: dict[str, pd.DataFrame]):
    """
    Grouped bar chart: languages on x-axis, one bar cluster per language,
    bars within cluster = models (hue), accuracy + asymmetric CI error bars.

    lang_data: dict of language -> Overall DataFrame (index = llm).
    """
    # Collect all models present across all languages, keep canonical order
    all_models = set()
    for df in lang_data.values():
        all_models.update(df.index.tolist())
    models = sort_models(list(all_models))

    languages = [l for l in LANGUAGES if l in lang_data]

    n_lang = len(languages)
    n_models = len(models)

    # Sizing: give each language group 1.4 inches, plus 2 inches for legend
    fig_width = n_lang * 1.7 + 2.5
    bar_group_width = 0.8          # fraction of each unit interval used by bars
    width = bar_group_width / n_models
    x = np.arange(n_lang)
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(fig_width, 3))

    for j, m in enumerate(models):
        color = MODEL_COLORS.get(m, "#AAAAAA")
        vals, lows, highs, xs = [], [], [], []

        for i, lang in enumerate(languages):
            df = lang_data[lang]
            if m not in df.index:
                continue
            vals.append(df.loc[m, "accuracy"] * 100)
            lows.append(df.loc[m, "CI_lower"] * 100)
            highs.append(df.loc[m, "CI_upper"] * 100)
            xs.append(x[i] + offsets[j])

        if not vals:
            continue

        ax.bar(
            xs, vals, width,
            color=color, edgecolor="black", linewidth=0.5,
            label=MODEL_DISPLAY_NAMES.get(m, m), zorder=3,
        )

        # Thinner, lighter CI bars so they don't dominate
        lower_err = np.array(vals) - np.array(lows)
        upper_err = np.array(highs) - np.array(vals)
        ax.errorbar(
            xs, vals,
            yerr=[lower_err, upper_err],
            fmt="none",
            capsize=2,
            ecolor="#333333",
            elinewidth=0.7,
            capthick=0.7,
            zorder=5,
        )



    ax.set_xticks(x)
    ax.set_xticklabels(
        [LANGUAGE_DISPLAY.get(l, l.capitalize()) for l in languages],
        fontsize=11,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.5, n_lang - 0.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend outside plot to the right, single column
    ax.legend(
        title="Model",
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        ncol=1,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])

    out = FIGURES_FOLDER / "Multilingual_accuracy_framed.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

def plot_multilingual_accuracy_subplots(
    lang_data: dict,
    save_path=None,
):
    """
    One subplot per language: models on x-axis, accuracy bars in uniform
    light blue with 95% CI error bars.

    Parameters
    ----------
    lang_data : dict[str, pd.DataFrame]
        language -> Overall DataFrame (index = llm).
    save_path : Path, optional
        Output PDF path. Defaults to FIGURES_FOLDER / "Multilingual_accuracy_subplots.pdf".
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from icd11_utils import LANGUAGES, MODEL_DISPLAY_NAMES, sort_models

    LANGUAGE_DISPLAY = {
        "english":  "English",
        "spanish":  "Spanish",
        "chinese":  "Chinese",
        "french":   "French",
        "japanese": "Japanese",
        "russian":  "Russian",
    }
    COLOR_CLOSED = "#9287C1" # "#A094E0" 
    COLOR_OPEN   = "#BCB6DD"   # lighter blue — open-source

    CLOSED_SOURCE_MODELS = { "gemini_25", "gpt_51", "claude_opus46"}

    # Canonical order
    languages = [l for l in LANGUAGES if l in lang_data]
    all_models = set()
    for df in lang_data.values():
        all_models.update(df.index.tolist())
    models = sort_models(list(all_models))
    bar_colors = [COLOR_CLOSED if m in CLOSED_SOURCE_MODELS else COLOR_OPEN for m in models]

    n_lang = len(languages)
    ncols = min(3, n_lang)
    nrows = int(np.ceil(n_lang / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.0, nrows * 3.5),
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    x = np.arange(len(models))
    width = 0.65

    for idx, lang in enumerate(languages):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        df = lang_data[lang]

        vals, ci_los, ci_his = [], [], []
        for m in models:
            if m in df.index:
                vals.append(df.loc[m, "accuracy"] * 100)
                ci_los.append(df.loc[m, "CI_lower"] * 100)
                ci_his.append(df.loc[m, "CI_upper"] * 100)
            else:
                vals.append(0)
                ci_los.append(0)
                ci_his.append(0)

        for i, m in enumerate(models):
            ax.bar(
                x[i], vals[i], width,
                color=bar_colors[i], edgecolor="black", linewidth=0.3, zorder=3,
            )

        lower_err = np.array(vals) - np.array(ci_los)
        upper_err = np.array(ci_his) - np.array(vals)
        ax.errorbar(
            x, vals,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=2.5,
            ecolor="#333333", elinewidth=0.7, capthick=0.7,
            zorder=5,
        )

        ax.set_title(
            LANGUAGE_DISPLAY.get(lang, lang.capitalize()),
            fontsize=11, fontweight="bold", pad=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(m, m) for m in models],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_ylim(0, 108)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10)

    # Hide unused subplots
    for idx in range(n_lang, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COLOR_OPEN, label="Open-Source", edgecolor="black"),
        mpatches.Patch(color=COLOR_CLOSED, label="Closed-Source", edgecolor="black"),
    ]
    fig.legend(
        handles=handles,
        fontsize=9, title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path is None:
        from run_plots import FIGURES_FOLDER
        save_path = FIGURES_FOLDER / "Multilingual_accuracy_subplots.pdf"
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_multilingual_per_model(
    lang_data: dict,
    save_path=None,
):
    """
    One subplot per model: languages on x-axis, accuracy bars in light blue
    with 95% CI error bars. English performance shown as a horizontal
    reference line with grey CI band spanning each subplot.

    Parameters
    ----------
    lang_data : dict[str, pd.DataFrame]
        language -> Overall DataFrame (index = llm).
    save_path : Path, optional
        Output PDF path.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    from icd11_utils import LANGUAGES, MODEL_DISPLAY_NAMES, sort_models

    LANGUAGE_DISPLAY = {
        "english":  "English",
        "spanish":  "Spanish",
        "chinese":  "Chinese",
        "french":   "French",
        "japanese": "Japanese",
        "russian":  "Russian",
    }

    BAR_COLOR = "#A094E0"

    # Canonical orders
    all_models = set()
    for df in lang_data.values():
        all_models.update(df.index.tolist())
    models = sort_models(list(all_models))

    # Non-English languages for the bars
    non_english = [l for l in LANGUAGES if l in lang_data and l != "english"]

    n_models = len(models)
    ncols = min(5, n_models)
    nrows = int(np.ceil(n_models / ncols))

    x = np.arange(len(non_english))
    width = 0.55

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.5, nrows * 3.0),
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # --- English reference line + CI band ---
        if "english" in lang_data and model in lang_data["english"].index:
            df_en = lang_data["english"]
            en_mean = df_en.loc[model, "accuracy"] * 100
            en_lo   = df_en.loc[model, "CI_lower"] * 100
            en_hi   = df_en.loc[model, "CI_upper"] * 100

            ax.axhspan(en_lo, en_hi, color="#D9D9D9", alpha=0.5, zorder=1)
            ax.axhline(en_mean, color="#666666", linewidth=1.2, linestyle="--", zorder=2)

        # --- Non-English bars ---
        vals, ci_los, ci_his = [], [], []
        for lang in non_english:
            df = lang_data[lang]
            if model in df.index:
                vals.append(df.loc[model, "accuracy"] * 100)
                ci_los.append(df.loc[model, "CI_lower"] * 100)
                ci_his.append(df.loc[model, "CI_upper"] * 100)
            else:
                vals.append(0)
                ci_los.append(0)
                ci_his.append(0)

        ax.bar(
            x, vals, width,
            color=BAR_COLOR, edgecolor="black", linewidth=0.3, zorder=3,
        )

        lower_err = np.array(vals) - np.array(ci_los)
        upper_err = np.array(ci_his) - np.array(vals)
        ax.errorbar(
            x, vals,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=2.5,
            ecolor="#333333", elinewidth=0.7, capthick=0.7,
            zorder=5,
        )

        ax.set_title(
            MODEL_DISPLAY_NAMES.get(model, model),
            fontsize=10, fontweight="bold", pad=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LANGUAGE_DISPLAY.get(l, l.capitalize()) for l in non_english],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_ylim(0, 108)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared legend
    handles = [
        mpatches.Patch(color=BAR_COLOR, label="Non-English", edgecolor="black"),
        plt.Line2D([0], [0], color="#666666", linewidth=1.2, linestyle="--", label="English Accuracy"),
        mpatches.Patch(facecolor="#D9D9D9", alpha=0.5, edgecolor="#BBBBBB", label="English 95% CI"),
    ]
    fig.legend(
        handles=handles,
        fontsize=9, title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path is None:
        from run_plots import FIGURES_FOLDER
        save_path = FIGURES_FOLDER / "Multilingual_accuracy_per_model.pdf"
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_accuracy_full_vs_essential(
    df_full: "pd.DataFrame",
    df_essential: "pd.DataFrame",
    path: str = None,
):
    """
    Single grouped bar chart: full-feature accuracy (dark) + essential-feature
    accuracy (light) side-by-side per model, both with 95% CIs.
    No clinicians. Models follow canonical MODEL_ORDER.

    Parameters
    ----------
    df_full : pd.DataFrame
        Overall sheet from the full-feature classification_metrics.xlsx.
    df_essential : pd.DataFrame
        Overall sheet from the essential-feature classification_metrics.xlsx.
    path : str or Path, optional
        Output path. Defaults to FIGURES_FOLDER_ABLATION / "accuracy_full_vs_essential.pdf".
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    from icd11_utils import MODEL_DISPLAY_NAMES, sort_models

    COLOR_FULL = "#5068A8"      # dark blue — full features
    COLOR_ESSENTIAL = "#C5D4F0" # light blue — essential features

    # Use models present in both DataFrames
    models_full = set(df_full.index.tolist())
    models_ess = set(df_essential.index.tolist())
    models = sort_models(list(models_full & models_ess))

    n = len(models)
    x = np.arange(n)
    width = 0.38

    # Extract values
    full_vals  = [df_full.loc[m, "accuracy"] * 100 for m in models]
    full_lower = [df_full.loc[m, "CI_lower"] * 100 for m in models]
    full_upper = [df_full.loc[m, "CI_upper"] * 100 for m in models]

    ess_vals  = [df_essential.loc[m, "accuracy"] * 100 for m in models]
    ess_lower = [df_essential.loc[m, "CI_lower"] * 100 for m in models]
    ess_upper = [df_essential.loc[m, "CI_upper"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(max(9, n * 1.0), 5))

    # Full-feature bars (left)
    ax.bar(
        x - width / 2, full_vals, width,
        color=COLOR_FULL, edgecolor="black", linewidth=0.3, zorder=3,
    )
    # Essential-feature bars (right)
    ax.bar(
        x + width / 2, ess_vals, width,
        color=COLOR_ESSENTIAL, edgecolor="black", linewidth=0.3, zorder=3,
    )

    # CI error bars — full
    lower_err = np.array(full_vals) - np.array(full_lower)
    upper_err = np.array(full_upper) - np.array(full_vals)
    ax.errorbar(
        x - width / 2, full_vals,
        yerr=[lower_err, upper_err],
        fmt="none", capsize=3,
        ecolor="black", elinewidth=0.9, zorder=5,
    )

    # CI error bars — essential
    lower_err = np.array(ess_vals) - np.array(ess_lower)
    upper_err = np.array(ess_upper) - np.array(ess_vals)
    ax.errorbar(
        x + width / 2, ess_vals,
        yerr=[lower_err, upper_err],
        fmt="none", capsize=3,
        ecolor="black", elinewidth=0.9, zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    full_patch = mpatches.Patch(
        color=COLOR_FULL, label="No Essential Features", edgecolor="black",
    )
    ess_patch = mpatches.Patch(
        color=COLOR_ESSENTIAL, label="With Essential Features", edgecolor="black",
    )
    ax.legend(
        handles=[full_patch, ess_patch],
        title="Feature Set", fontsize=9, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0, framealpha=0.9, edgecolor="black",
    )

    plt.tight_layout()
    if path is None:
        from run_plots import FIGURES_FOLDER_ABLATION
        path = FIGURES_FOLDER_ABLATION / "accuracy_full_vs_essential.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def plot_ablation_sensitivity(
    df_english: "pd.DataFrame",
    ablation_path: "Path",
    save_path: "Path" = None,
):
    """
    One subplot per model: original accuracy (from df_english) + 6 ablation
    conditions (prompt_v1/v2/v3, paraphrase_low/medium/high) with 95% CIs.

    Parameters
    ----------
    df_english : pd.DataFrame
        Overall sheet from English classification_metrics.xlsx (index = llm).
    ablation_path : Path
        Path to ablation_sensitivity_accuracy_wilson.xlsx.
    save_path : Path, optional
        Output PDF path. Defaults to FIGURES_FOLDER_ABLATION / "ablation_sensitivity.pdf".
    """
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from icd11_utils import MODEL_COLORS, MODEL_DISPLAY_NAMES, sort_models

    # ------------------------------------------------------------------
    # Load & parse ablation data
    # ------------------------------------------------------------------
    df_abl = pd.read_excel(ablation_path, sheet_name="Ablation Results")
    df_abl.columns = df_abl.columns.str.strip()
    df_abl = df_abl.set_index("llm")

    # Parse "0.6279 [0.4786, 0.7562]" → (mean, ci_lower, ci_upper)
    def _parse_cell(cell):
        if pd.isna(cell):
            return np.nan, np.nan, np.nan
        s = str(cell).strip()
        m = re.match(r"([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", s)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
        # Try plain number
        try:
            v = float(s)
            return v, np.nan, np.nan
        except ValueError:
            return np.nan, np.nan, np.nan

    ablation_cols = [c for c in df_abl.columns]  # prompt_v1, prompt_v2, ...

    # ------------------------------------------------------------------
    # Build tidy structure: model → list of (condition, mean, ci_lo, ci_hi)
    # ------------------------------------------------------------------
    models = sort_models(
        [m for m in df_abl.index.tolist() if m in df_english.index]
    )

    # Condition display names and colours
    CONDITION_LABELS = {
        "original":         "Original",
        "prompt_v1":        "Prompt\nLow",
        "prompt_v2":        "Prompt\nMedium",
        "prompt_v3":        "Prompt\nHigh",
        "paraphrase_low":   "Vignette\nLow",
        "paraphrase_medium":"Vignette\nMedium",
        "paraphrase_high":  "Vignette\nHigh",
    }
    CONDITIONS = list(CONDITION_LABELS.keys())

    COND_COLORS = {
        "original":          "#808080",
        "prompt_v1":         "#C5D4F0",
        "prompt_v2":         "#7B96D4",
        "prompt_v3":         "#5068A8",
        "paraphrase_low":    "#BEF0D5",
        "paraphrase_medium": "#83B69B",
        "paraphrase_high":   "#22A37A",
    }

    # ------------------------------------------------------------------
    # Plot: one subplot per model
    # ------------------------------------------------------------------
    n_models = len(models)
    n_conds = len(CONDITIONS)
    ncols = min(5, n_models)
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 3.5),
        sharey=True,
    )
    axes = np.atleast_2d(axes)  # ensure 2D even for single row

    x = np.arange(n_conds)
    width = 0.65

    for idx, model in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        means, ci_los, ci_his = [], [], []

        for cond in CONDITIONS:
            if cond == "original":
                mean_val = df_english.loc[model, "accuracy"]
                ci_lo = df_english.loc[model, "CI_lower"]
                ci_hi = df_english.loc[model, "CI_upper"]
            else:
                mean_val, ci_lo, ci_hi = _parse_cell(
                    df_abl.loc[model, cond] if cond in df_abl.columns else np.nan
                )
            means.append(mean_val * 100)
            ci_los.append(ci_lo * 100 if not np.isnan(ci_lo) else mean_val * 100)
            ci_his.append(ci_hi * 100 if not np.isnan(ci_hi) else mean_val * 100)

        colors = [COND_COLORS[c] for c in CONDITIONS]

        bars = ax.bar(x, means, width, color=colors, edgecolor="black", linewidth=0.3, zorder=3)

        # Highlight the original bar
        bars[0].set_edgecolor("black")
        bars[0].set_linewidth(0.8)

        # CI error bars
        lower_err = np.array(means) - np.array(ci_los)
        upper_err = np.array(ci_his) - np.array(means)
        ax.errorbar(
            x, means,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=2.5,
            ecolor="#333333", elinewidth=0.7, capthick=0.7,
            zorder=5,
        )

        ax.set_title(
            MODEL_DISPLAY_NAMES.get(model, model),
            fontsize=10, fontweight="bold", pad=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in CONDITIONS],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylim(0, 105)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=9)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COND_COLORS[c], label=CONDITION_LABELS[c].replace("\n", " "), edgecolor="black")
        for c in CONDITIONS
    ]
    fig.legend(
        handles=handles,
        title="Condition", fontsize=8, title_fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(CONDITIONS),
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path is None:
        from run_plots import FIGURES_FOLDER_ABLATION
        save_path = FIGURES_FOLDER_ABLATION / "ablation_sensitivity.pdf"
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_ablation_sensitivity_refline(
    df_english: "pd.DataFrame",
    ablation_path: "Path",
    save_path: "Path" = None,
):
    """
    Variant: original accuracy shown as a horizontal reference line with a
    grey CI band spanning the full subplot. Only the 6 ablation conditions
    are drawn as bars (prompt cluster + vignette cluster).
 
    Parameters
    ----------
    df_english : pd.DataFrame
        Overall sheet from English classification_metrics.xlsx (index = llm).
    ablation_path : Path
        Path to ablation_sensitivity_accuracy_wilson.xlsx.
    save_path : Path, optional
        Output PDF path. Defaults to FIGURES_FOLDER_ABLATION / "ablation_sensitivity_refline.pdf".
    """
    import re
 
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
 
    from icd11_utils import MODEL_DISPLAY_NAMES, sort_models
 
    # ------------------------------------------------------------------
    # Load & parse ablation data
    # ------------------------------------------------------------------
    df_abl = pd.read_excel(ablation_path, sheet_name="Ablation Results")
    df_abl.columns = df_abl.columns.str.strip()
    df_abl = df_abl.set_index("llm")
 
    def _parse_cell(cell):
        if pd.isna(cell):
            return np.nan, np.nan, np.nan
        s = str(cell).strip()
        m = re.match(r"([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", s)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
        try:
            v = float(s)
            return v, np.nan, np.nan
        except ValueError:
            return np.nan, np.nan, np.nan
 
    models = sort_models(
        [m for m in df_abl.index.tolist() if m in df_english.index]
    )
 
    # Only the 6 ablation conditions (no "original")
    ABLATION_COLS = [
        "prompt_v1", "prompt_v2", "prompt_v3",
        "paraphrase_low", "paraphrase_medium", "paraphrase_high",
    ]
    CONDITION_LABELS = {
        "prompt_v1":        "Prompt\nLow",
        "prompt_v2":        "Prompt\nMedium",
        "prompt_v3":        "Prompt\nHigh",
        "paraphrase_low":   "Vignette\nLow",
        "paraphrase_medium":"Vignette\nMedium",
        "paraphrase_high":  "Vignette\nHigh",
    }
    COND_COLORS = {
        "prompt_v1":         "#C5D4F0",
        "prompt_v2":         "#7B96D4",
        "prompt_v3":         "#5068A8",
        "paraphrase_low":    "#BEF0D5",
        "paraphrase_medium": "#6AC193",
        "paraphrase_high":   "#248062",
    }
 
    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    n_models = len(models)
    ncols = min(5, n_models)
    nrows = int(np.ceil(n_models / ncols))
 
    # Grouped x: prompt cluster | gap | vignette cluster
    gap = 0.45
    width = 0.65
    x = np.array([
        0, 1, 2,                            # Prompt Low / Medium / High
        3 + gap, 4 + gap, 5 + gap,          # Vignette Low / Medium / High
    ])
 
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.0, nrows * 3.5),
        sharey=True,
    )
    axes = np.atleast_2d(axes)
 
    for idx, model in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
 
        # --- Original as reference line + CI band ---
        orig_mean = df_english.loc[model, "accuracy"] * 100
        orig_lo   = df_english.loc[model, "CI_lower"] * 100
        orig_hi   = df_english.loc[model, "CI_upper"] * 100
 
        ax.axhspan(orig_lo, orig_hi, color="#D9D9D9", alpha=0.5, zorder=1)
        ax.axhline(orig_mean, color="#666666", linewidth=1.2, linestyle="--", zorder=2)
 
        # --- Ablation bars ---
        means, ci_los, ci_his = [], [], []
        for cond in ABLATION_COLS:
            mean_val, ci_lo, ci_hi = _parse_cell(
                df_abl.loc[model, cond] if cond in df_abl.columns else np.nan
            )
            means.append(mean_val * 100)
            ci_los.append(ci_lo * 100 if not np.isnan(ci_lo) else mean_val * 100)
            ci_his.append(ci_hi * 100 if not np.isnan(ci_hi) else mean_val * 100)
 
        colors = [COND_COLORS[c] for c in ABLATION_COLS]
        ax.bar(x, means, width, color=colors, edgecolor="black", linewidth=0.3, zorder=3)
 
        lower_err = np.array(means) - np.array(ci_los)
        upper_err = np.array(ci_his) - np.array(means)
        ax.errorbar(
            x, means,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=2.5,
            ecolor="#333333", elinewidth=0.7, capthick=0.7,
            zorder=5,
        )
 
        ax.set_title(
            MODEL_DISPLAY_NAMES.get(model, model),
            fontsize=10, fontweight="bold", pad=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in ABLATION_COLS],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylim(0, 105)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
 
        if col == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=9)
 
    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)
 
    # Shared legend: ablation bars + reference line/band
    handles = [
        mpatches.Patch(color=COND_COLORS[c], label=CONDITION_LABELS[c].replace("\n", " "), edgecolor="black")
        for c in ABLATION_COLS
    ]
    handles.append(plt.Line2D([0], [0], color="#666666", linewidth=1.2, linestyle="--", label="Original Accuracy"))
    handles.append(mpatches.Patch(facecolor="#D9D9D9", alpha=0.5, edgecolor="#BBBBBB", label="Original 95% CI"))
 
    fig.legend(
        handles=handles,
        fontsize=9, title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )
 
    plt.tight_layout(rect=[0, 0.08, 1, 1])
 
    if save_path is None:
        from run_plots import FIGURES_FOLDER_ABLATION
        save_path = FIGURES_FOLDER_ABLATION / "ablation_sensitivity_refline.pdf"
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

# ---------------------------------------------------------------------------
# Plot: English Top-N Accuracy
# ---------------------------------------------------------------------------

def plot_english_topn_accuracy(topn_path: Path, save_path: Path = None):
    """
    Grouped bar chart: models on x-axis, hue = Top-1 / Top-2 / Top-3 accuracy,
    with 95% Wilson CI error bars.

    Parameters
    ----------
    topn_path : Path
        Path to topn_accuracy.xlsx (output of build_topn_sheet).
    save_path : Path, optional
        Output PDF path. Defaults to FIGURES_FOLDER / "English_topn_accuracy.pdf".
    """
    if not topn_path.exists():
        print(f"  WARNING: {topn_path} not found, skipping Top-N plot.")
        return

    df = pd.read_excel(topn_path, sheet_name="TopN_Accuracy", index_col="llm")
    models = sort_models(df.index.tolist())

    COLOR_TOP1 = "#5068A8"   # dark blue
    COLOR_TOP2 = "#89ABEB"   # medium blue
    COLOR_TOP3 = "#C5D4F0"   # light blue

    metrics = {
        "Top-1": ("top1_accuracy", "top1_CI_lower", "top1_CI_upper", COLOR_TOP1),
        "Top-2": ("top2_accuracy", "top2_CI_lower", "top2_CI_upper", COLOR_TOP2),
        "Top-3": ("top3_accuracy", "top3_CI_lower", "top3_CI_upper", COLOR_TOP3),
    }

    n = len(models)
    n_metrics = len(metrics)
    x = np.arange(n)
    width = 0.25
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    for j, (label, (col_acc, col_lo, col_hi, color)) in enumerate(metrics.items()):
        vals  = [df.loc[m, col_acc] * 100 for m in models]
        lows  = [df.loc[m, col_lo]  * 100 for m in models]
        highs = [df.loc[m, col_hi]  * 100 for m in models]

        ax.bar(
            x + offsets[j], vals, width,
            color=color, edgecolor="black", linewidth=0.7,
            label=label, zorder=3,
        )

        lower_err = np.array(vals) - np.array(lows)
        upper_err = np.array(highs) - np.array(vals)
        ax.errorbar(
            x + offsets[j], vals,
            yerr=[lower_err, upper_err],
            fmt="none", capsize=3,
            ecolor="#333333", elinewidth=1.0, capthick=1.0,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models],
        rotation=45, ha="right", fontsize=10,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Top-N", fontsize=11, title_fontsize=11,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0, framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_FOLDER / "English_topn_accuracy.pdf"
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load English metrics
    df_english = load_overall_sheet("english")
    if df_english is None:
        raise FileNotFoundError("English classification_metrics.xlsx not found.")

    # Load clinician data
    clinician = load_clinicians()

    print("Generating Plot 1a: English accuracy + kappa (side-by-side panels) ...")
    plot_english_accuracy_kappa(df_english, clinician)
    plot_english_accuracy_kappa2(df_english, clinician)

    print("Generating Plot 1b: English accuracy + kappa (side-by-side panels) ...")
    plot_english_accuracy_kappa_refline(df_english, clinician)

    print("Generating Plot 1c: English accuracy + kappa (2-colour single panel) ...")
    plot_english_accuracy_kappa_2color(df_english, clinician)

    print("Generating Plot 2a: English sensitivity / specificity / F1 ...")
    plot_english_sens_spec_f1_weighted(df_english, clinician)

    print("Generating Plot 2b: English sensitivity / specificity / F1 ...")
    plot_english_sens_spec_f1_macro(df_english, clinician)

    print("Generating Plot 2c: English sensitivity / precision / F1 ...")
    plot_english_sens_prec_f1_weighted(df_english, clinician)

    print("Generating Plot 2d: English macro sensitivity / precision / weighted F1 ...")
    plot_english_sens_prec_f1_macro(df_english, clinician)

    plot_english_sens_spec_prec_f1_weighted(df_english, clinician)

    print("Generating Plot 3a: English accuracy per category ...")
    plot_english_per_category()

    print("Generating Plot 3b: English accuracy per category per subplot...")
    plot_english_per_category_subplots(
        METRICS_FOLDER,
        CLINICIAN_CSV,
        FIGURES_FOLDER,
    )


    # Load all languages for multilingual plot (English + others)
    print("Loading multilingual results ...")
    lang_data = {}
    for lang in LANGUAGES:
        df = load_overall_sheet(lang)
        if df is not None:
            lang_data[lang] = df

    print("Generating Plot 3a: Multilingual accuracy ...")
    plot_multilingual_accuracy(lang_data)

    print("Generating Plot 3b: Multilingual accuracy as sub-plots...")
    plot_multilingual_accuracy_subplots(
    lang_data,
    FIGURES_FOLDER / "Multilingual_accuracy_subplots.pdf",
    )
    print("Generating Plot 3b: Multilingual accuracy as sub-plots per model...")
    plot_multilingual_per_model(
    lang_data,
    FIGURES_FOLDER / "Multilingual_accuracy_per_model.pdf",
    )

    print("Generating Plot 4: Essential features ...")
    path_essential = RESULTS_FOLDER / "ablation" / "_results" / "essential_features_metrics.xlsx"
    df_essential = load_overall_sheet("english", path_essential)
    plot_english_accuracy_kappa_2color(df_english, clinician, FIGURES_FOLDER_ABLATION / "English_accuracy_kappa_2color_essentials.pdf")

    print("Generating Plot 5: Essential features compared to full accuracy")
    plot_accuracy_full_vs_essential(
        df_english,
        df_essential,
        FIGURES_FOLDER_ABLATION / "accuracy_full_vs_essential.pdf",
    )   

    ablation_wilson_path = RESULTS_FOLDER / "ablation" / "_results" / "ablation_sensitivity_accuracy_wilson.xlsx"
    plot_ablation_sensitivity(
        df_english,
        ablation_wilson_path,
        FIGURES_FOLDER_ABLATION / "ablation_sensitivity.pdf",
    )
    plot_ablation_sensitivity_refline(
        df_english,
        ablation_wilson_path,
        FIGURES_FOLDER_ABLATION / "ablation_sensitivity_refline.pdf",
    )

    print("Generating Plot 6: Top-N accuracy plot")
    plot_english_topn_accuracy(
        METRICS_FOLDER / "english" / "topn_accuracy.xlsx",
        FIGURES_FOLDER / "English_topn_accuracy.pdf",
    )

    print("\nDone. All figures saved to:", FIGURES_FOLDER)