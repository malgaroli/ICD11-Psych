"""
icd11_utils.py
==============
Shared utilities for the ICD-11 LLM diagnostic benchmarking project.

Imported by:
    - run_classification_metrics.py
    - run_essential_features.py
    - run_sensitivity_ablation.py
    - run_contamination_summary.py
"""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "mistral_7B": "Mistral 7B",
    "llama31_8B": "Llama 3.1 8B",
    "gemma3": "Gemma 3",
    "Qwen25_32B": "Qwen 32B",
    "deepseek_70B": "DeepSeek R1 70B",
    "llama33_70B": "Llama 3.3 70B",
    "mistral_large": "Mistral Large",
    "gemini_25": "Gemini 2.5 Pro",
    "gpt_51": "GPT-5.1",
    "claude_opus46": "Claude Opus 4.6"
}

# Canonical display order — used by sort_models() to order rows/bars consistently.
# Any model not listed here will be appended alphabetically at the end.
MODEL_ORDER = [
    "mistral_7B",
    "llama31_8B",
    "gemma3",
    "Qwen25_32B",
    "deepseek_70B",
    "llama33_70B",
    "mistral_large",
    "gemini_25",       # March 2025
    "gpt_51",          # November 2025
    "claude_opus46",   # February 2026
]

LANGUAGES = ["english" ]#, "spanish", "chinese", "french", "japanese", "russian"]

LABEL_MAP = {
    #english
    "No Diagnosis; normal reaction to stressful event(s)": "No Diagnosis",
    "No Diagnosis; symptoms and behaviours are within normal limits": "No Diagnosis",
    "Adjustment Disorder with Depressed Mood": "Adjustment Disorder",
    "Adjustment Disorder with Depressive Symptoms": "Adjustment Disorder",
    "Adjustment Disorder with Depressive and Anxious Features": "Adjustment Disorder",
    "10. Separation Anxiety Disorder": "Separation Anxiety Disorder",
    "Bipolar II Disorder" : "Bipolar Type II Disorder",

    
    #french
    "Trouble danxiété généralisée" : "Trouble d'anxiété généralisée",
    "Trouble d'anxiété généralisé" : "Trouble d'anxiété généralisée",
    "Trouble danxiété de séparation" : "Trouble d'anxiété de séparation",
    "Trouble danxiété sociale" : "Trouble d'anxiété sociale",
    "Trouble anxieux social" : "Trouble d'anxiété sociale",
    
    # russian
    "Единичный эпизод депрессивного расстройства": "Единичный эпизод депрессивного расстройства",
    "Единичный эпизод депрессивный расстройство": "Единичный эпизод депрессивного расстройства",  # grammatical error in model output
    "Агорафобия И Паническое расстройство": "Агорафобия И Паническое расстройство",
    "Социальное тревожное расстройство": "Социальное тревожное расстройство",
    "Социальное тревожное расстройство И Паническое расстройство": "Социальное тревожное расстройство И Паническое расстройство",
    "Сепарационное тревожное расстройство и Паническое расстройство": "Сепарационное тревожное расстройство И Паническое расстройство",  # lowercase и → И
    "Социальное тревожное расстройство и Паническое расстройство (социальное тревожное расстройство с паническими атаками)": "Социальное тревожное расстройство И Паническое расстройство",  # verbose + lowercase и
    # Not in canonical list — leave as-is:
    "Депрессивное расстройство, острое": "Депрессивное расстройство, острое",
    "Пролонгированное горе": "Пролонгированное горе",
    "Общее социальное и сепарационное тревожное расстройство": "Общее социальное и сепарационное тревожное расстройство",

    # spanish
    "Trastorno de ansiedad generalizada.": "Trastorno de ansiedad generalizada",  # trailing period
    "Trastorno de ansiedad generalizada": "Trastorno de Ansiedad Generalizada",  # lowercase
    "Trastorno por estrés postraumático (TEPT)": "Trastorno por estrés postraumático",  # parenthetical acronym
    "[diagnóstico]": "NO_MATCH_FOUND",  # placeholder, model failed to produce a diagnosis
    # Not in canonical list — leave as-is:
    "Trastorno Delirante": "Trastorno Delirante",
    "Trastorno Depresivo Postparto": "Trastorno Depresivo Postparto",
    "Trastorno Depresivo Postpartum": "Trastorno Depresivo Postpartum",
    "Trastorno Depresivo Puerperal": "Trastorno Depresivo Puerperal",
    "Trastorno por estrés agudo": "Trastorno por estrés agudo",
    "Trastorno de ansiedad por separación y Trastorno de ansiedad generalizada": "Trastorno de ansiedad por separación y Trastorno de ansiedad generalizada",

    # japanese
    "一般性不安障害": "全般性不安障害",  # synonym for generalized
    "广场恐惧症 以及 Q2 恐慌障碍": "広場恐怖症 および Q2 パニック障害",  # Simplified Chinese → Japanese
    "気分変調性障害": "気分変調性障害",  # exact match
    "遅延性悲嘆障害": "遷延性悲嘆障害",  # kanji variant, same diagnosis
    "心的外傷後ストレス障害 (PTSD)": "心的外傷後ストレス障害",  # parenthetical acronym
    "心的外傷後ストレス障害（PTSD）": "心的外傷後ストレス障害",  # full-width parentheses
    "心的外傷後ストレス障害 ( PTSD )": "心的外傷後ストレス障害",  # spaced parenthetical
    "複雑性心的外傷後ストレス障害 (Complex Post-Traumatic Stress Disorder, C-PTSD)": "複雑性心的外傷後ストレス障害",  # English annotation
    "長期にわたる悲嘆障害（Prolonged Grief Disorder）": "遷延性悲嘆障害",  # descriptive paraphrase + English
    # Not in canonical list — leave as-is:
    "持続性抑うつ障害": "持続性抑うつ障害",
    "反応性抑うつ障害": "反応性抑うつ障害",
    "抑うつ障害": "抑うつ障害",
    "産後うつ病": "産後うつ病",
    "複合性抑うつ障害": "複合性抑うつ障害",
    "ストレス関連障害群": "ストレス関連障害群",

    # chinese
    "[分离性焦虑障碍]": "分离性焦虑障碍",  # brackets
    "分离性焦虑障碍和惊恐障碍": "分离性焦虑障碍 并 惊恐障碍",  # 和 → 并
    "社交焦虑障碍和惊恐障碍": "社交焦虑障碍 并 惊恐障碍",  # 和 → 并
}

# Colour palettes (shared across plotting scripts)
COLORS = [
    "#394C89", "#86C3FF", "#9771CC", "#DE92AB",
    "#C25A7D", "#6F7A8A", "#B8C0CE", "#6CDFD5",
]
COLORS_3 = ["#394C89", "#93A4DC", "#BFBFBF"]
COLORS_LNG = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#9C9C9C"]
 
# Per-model colour palette — keyed to MODEL_ORDER for consistency across all plots.
# Any model not listed falls back to "#AAAAAA" in plotting code.
MODEL_COLORS = {
    "mistral_7B":    "#6C81C6",
    "llama31_8B":    "#86C3FF",
    "gemma3":        "#B597DF",
    "Qwen25_32B":    "#E0A4B8",
    "deepseek_70B":  "#A95873",
    "llama33_70B":   "#6F7A8A",
    "mistral_large": "#B8C0CE",
    "gemini_25":     "#7CCDB1",
    "gpt_51":        "#238F60",
    "claude_opus46": "#B8DED0",
    "Clinicians":    "#E9C46A",
}

# Canonical label orders for English confusion matrices (by category)
ENGLISH_CM_LABEL_ORDER = {
    "Anxiety": [
        "Generalized Anxiety Disorder",
        "Generalized Anxiety Disorder AND Panic Disorder",
        "Panic Disorder",
        "Agoraphobia",
        "Agoraphobia AND Panic Disorder",
        "Specific Phobia",
        "Specific Phobia AND Panic Disorder",
        "Social Anxiety Disorder",
        "Social Anxiety Disorder AND Panic Disorder",
        "Separation Anxiety Disorder",
        "Separation Anxiety Disorder AND Panic Disorder",
        "Selective Mutism",
        "Other Anxiety and Fear-Related Disorder",
        "Unspecified Anxiety and Fear-Related Disorder",
        "A different diagnosis",
        "No Diagnosis",
    ],
    "Mood": [
        "Single Episode Depressive Disorder",
        "Recurrent Depressive Disorder",
        "Dysthymic Disorder",
        "Mixed Depressive and Anxiety Disorder",
        "Bipolar Type I Disorder",
        "Bipolar Type II Disorder",
        "Cyclothymic Disorder",
        "Other Mood Disorder",
        "Generalized Anxiety Disorder",
        "Prolonged Grief Disorder",
        "No Diagnosis",
        "Adjustment Disorder",
        "A different diagnosis not listed above",
    ],
    "Stress": [
        "Post-Traumatic Stress Disorder (PTSD)",
        "Complex Post-Traumatic Stress Disorder (CPTSD)",
        "Prolonged Grief Disorder",
        "No Diagnosis",
        "Adjustment Disorder",
        "Other Disorder Specifically Associated with Stress",
        "Acute Stress Reaction",
        "A diagnosis from a different diagnostic area (e.g. mood disorders, psychotic disorders, personality disorders)",
    ],
}

ANXIETY_X = {
    "GAD":                                                       "Q1",
    "Generalized Anxiety Disorder AND Panic Disorder":           "Q1 & Q2",
    "Panic Disorder":                                            "Q2",
    "Agoraphobia":                                               "Q3",
    "Agoraphobia AND Panic Disorder":                            "Q3 & Q2",
    "Specific Phobia":                                           "Q4",
    "Specific Phobia AND Panic Disorder":                        "Q4 & Q2",
    "Social Anxiety Disorder":                                   "Q5",
    "Social Anxiety Disorder AND Panic Disorder":                "Q5 & Q2",
    "Separation Anxiety Disorder":                               "Q6",
    "Separation Anxiety Disorder AND Panic Disorder":            "Q6 & Q2",
    "Selective Mutism":                                          "Q7",
    "Other Anxiety and Fear-Related Disorder":                   "Q8",
    "Unspecified Anxiety and Fear-Related Disorder":             "Q9",
    "A different diagnosis":                                     "Other",
    "No Diagnosis":                                              "None",
}

ANXIETY_Y = {
    "GAD":                                                       "Q1 (GAD)",
    "Generalized Anxiety Disorder AND Panic Disorder":           "Q1 (GAD) & Q2 (Panic)",
    "Panic Disorder":                                            "Q2 (Panic)",
    "Agoraphobia":                                               "Q3 (Agoraphobia)",
    "Agoraphobia AND Panic Disorder":                            "Q3 (Agoraphobia) & Q2 (Panic)",
    "Specific Phobia":                                           "Q4 (Specific Phobia)",
    "Specific Phobia AND Panic Disorder":                        "Q4 (Specific Phobia) & Q2 (Panic)",
    "Social Anxiety Disorder":                                   "Q5 (Social Anxiety Disorder)",
    "Social Anxiety Disorder AND Panic Disorder":                "Q5 (Social Anxiety Disorder) & Q2 (Panic)",
    "Separation Anxiety Disorder":                               "Q6 (Separation)",
    "Separation Anxiety Disorder AND Panic Disorder":            "Q6 (Separation) & Q2 (Panic)",
    "Selective Mutism":                                          "Q7 (Selective Mutism)",
    "Other Anxiety and Fear-Related Disorder":                   "Q8 (Other)",
    "Unspecified Anxiety and Fear-Related Disorder":             "Q9 (Unspecified)",
    "A different diagnosis":                                     "Other",
    "No Diagnosis":                                              "None (No Diagnosis)",
}

MOOD_X = {
    "Single Episode Depressive Disorder":                        "L1",
    "Recurrent Depressive Disorder":                             "L2",
    "Dysthymic Disorder":                                        "L3",
    "Mixed Depressive and Anxiety Disorder":                     "L4",
    "Bipolar I":                                                 "L5",
    "Bipolar II":                                                "L6",
    "Cyclothymic Disorder":                                      "L7",
    "Other Mood Disorder":                                       "L8",
    "GAD":                                                       "N1",
    "PGD":                                                       "P1",
    "No Diagnosis":                                              "None",
    "Adjustment Disorder":                                       "P2",
    "Different Diagnosis":                                       "Other",
}

MOOD_Y = {
    "Single Episode Depressive Disorder":                        "L1 (Single Episode Depressive Disorder)",
    "Recurrent Depressive Disorder":                             "L2 (Recurrent Depressive Disorder)",
    "Dysthymic Disorder":                                        "L3 (Dysthymia)",
    "Mixed Depressive and Anxiety Disorder":                     "L4 (Mixed Dep/Anx)",
    "Bipolar I":                                                 "L5 (Bipolar I)",
    "Bipolar II":                                                "L6 (Bipolar II)",
    "Cyclothymic Disorder":                                      "L7 (Cyclothymia)",
    "Other Mood Disorder":                                       "L8 (Other Mood)",
    "GAD":                                                       "N1 (GAD)",
    "PGD":                                                       "P1 (PGD)",
    "No Diagnosis":                                              "None (No Diagnosis)",
    "Adjustment Disorder":                                       "P2 (Adjustment Disorder)",
    "Different Diagnosis":                                       "Other",
}

STRESS_X = {
    "PTSD":                                                                                         "A1",
    "C-PTSD":                                                                                       "A2",
    "PGD":                                                                                          "A3",
    "No Diagnosis":                                                                                 "None",
    "Adjustment Disorder":                                                                          "A4",
    "Different Stress-related Diagnosis":                                                           "A5",
    "Acute Stress Reaction":                                                                        "A6",
    "Different Diagnosis":                                                                          "Other",
}

STRESS_Y = {
    "PTSD":                                                                                         "A1 (PTSD)",
    "C-PTSD":                                                                                       "A2 (C-PTSD)",
    "PGD":                                                                                          "A3 (PGD)",
    "No Diagnosis":                                                                                 "None (No Diagnosis)",
    "Adjustment Disorder":                                                                          "A4 (Adjustment Disorder)",
    "Different Stress-related Diagnosis":                                                           "A5 (Other Stress-related Diagnosis)",
    "Acute Stress Reaction":                                                                        "A6 (Acute Stress Reaction)",
    "Different Diagnosis":                                                                          "Other",
    

}

# Combine into single lookup by category
CM_X_AXIS_MAP = {"Anxiety": ANXIETY_X, "Mood": MOOD_X, "Stress": STRESS_X}
CM_Y_AXIS_MAP = {"Anxiety": ANXIETY_Y, "Mood": MOOD_Y, "Stress": STRESS_Y}
# ---------------------------------------------------------------------------
# Model ordering
# ---------------------------------------------------------------------------

def sort_models(models: list[str]) -> list[str]:
    """
    Sort a list of model keys according to MODEL_ORDER.
    Any model not in MODEL_ORDER is appended alphabetically at the end.
    """
    order_index = {m: i for i, m in enumerate(MODEL_ORDER)}
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(m for m in models if m not in order_index)
    return known + unknown


def reindex_by_model_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a DataFrame whose index is model keys so rows follow MODEL_ORDER.
    Rows not in MODEL_ORDER are appended alphabetically at the end.
    """
    ordered = sort_models(df.index.tolist())
    return df.reindex(ordered)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def consolidate_label(label: str) -> str:
    """Map a label to its canonical form via LABEL_MAP."""
    return LABEL_MAP.get(label, label)


def extract_top1_prediction(model_diagnoses_str) -> str:
    """Extract the first (top-1) diagnosis from the Model_Diagnoses column."""
    try:
        diags = ast.literal_eval(model_diagnoses_str)
        if isinstance(diags, list) and len(diags) > 0:
            return diags[0]
    except (ValueError, SyntaxError):
        pass
    return str(model_diagnoses_str).strip()

# ---------------------------------------------------------------------------
# Clinician consensus loading
# ---------------------------------------------------------------------------

def load_clinician_consensus(path: Path) -> pd.Series:
    """
    Load clinician consensus labels from the detailed ratings Excel file.
    
    Returns a Series indexed by Vignette_ID with the consensus label,
    excluding any vignettes where the consensus is blank (tie).
    """
    df = pd.read_csv(path, index_col="Vignette")
    df.columns = df.columns.str.strip()
    consensus = df["Clinician Consensus"].dropna().str.strip()
    # Drop empty strings too
    consensus = consensus[consensus != ""]
    # Apply same label consolidation
    consensus = consensus.apply(consolidate_label)
    return consensus

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(files: list[Path], llm_part_index: int = -3) -> pd.DataFrame:
    """
    Load a list of result CSVs, strip column whitespace, and tag each row
    with the model name extracted from the file path.

    Args:
        files:          List of Path objects pointing to *results*.csv files.
        llm_part_index: Which path part contains the model name (default -3).

    Returns:
        Combined DataFrame with an added 'llm' column.
    """
    all_data = []
    for file in files:
        df = pd.read_csv(file, index_col="Vignette_ID")
        df.columns = df.columns.str.strip()
        df["llm"] = file.parts[llm_part_index]
        all_data.append(df)
    combined = pd.concat(all_data)
    return combined


def apply_label_corrections(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Predicted_Label column (top-1 prediction) and apply LABEL_MAP
    consolidation to both Ground_Truth_Label and Predicted_Label.
    """
    combined["Predicted_Label"] = (
        combined["Model_Diagnoses"].apply(extract_top1_prediction)
    )
    combined["Ground_Truth_Label"] = (
        combined["Ground_Truth_Label"].apply(consolidate_label)
    )
    combined["Predicted_Label"] = (
        combined["Predicted_Label"].apply(consolidate_label)
    )
    return combined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_per_class_sens_spec(
    y_true, y_pred, labels
) -> tuple[pd.DataFrame, float, float, float, float, float, float]:
    """
    Compute per-class sensitivity, specificity, and precision (one-vs-rest).

    Returns:
        per_class_df:   DataFrame with columns label, support, sensitivity, specificity, precision.
        macro_sens:     Macro (unweighted) mean sensitivity.
        macro_spec:     Macro (unweighted) mean specificity.
        macro_prec:     Macro (unweighted) mean precision.
        weighted_sens:  Support-weighted mean sensitivity.
        weighted_spec:  Support-weighted mean specificity.
        weighted_prec:  Support-weighted mean precision.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_classes = len(labels)
    supports = cm.sum(axis=1)
    total = cm.sum()

    rows = []
    sensitivities = np.zeros(n_classes)
    specificities = np.zeros(n_classes)
    precisions = np.zeros(n_classes)

    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        sensitivities[i] = sens
        specificities[i] = spec
        precisions[i] = prec

        rows.append({
            "label": labels[i],
            "support": int(supports[i]),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
        })

    macro_sens = float(np.mean(sensitivities))
    macro_spec = float(np.mean(specificities))
    macro_prec = float(np.mean(precisions))
    weighted_sens = float(np.average(sensitivities, weights=supports))
    weighted_spec = float(np.average(specificities, weights=supports))
    weighted_prec = float(np.average(precisions, weights=supports))

    return (
        pd.DataFrame(rows),
        macro_sens, macro_spec, macro_prec,
        weighted_sens, weighted_spec, weighted_prec,
    )



def compute_kappa_ci(y_true, y_pred, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Compute Cohen's kappa with asymptotic normal 95% CI.

    The standard error is derived from the confusion matrix following
    Fleiss, Cohen & Everitt (1969). Returns (kappa, ci_lower, ci_upper).
    """
    kappa = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    p_o = np.diag(cm).sum() / n                        # observed agreement
    p_e = (row_sums * col_sums).sum() / n**2           # expected agreement

    # Asymptotic SE (Fleiss et al.)
    # Components needed for the variance formula
    # Asymptotic SE (standard form, see Fleiss 1981 / Cohen 1960):
    se = np.sqrt((p_o * (1 - p_o)) / (n * (1 - p_e) ** 2))

    z = 1.96  # 95% CI
    ci_lower = kappa - z * se
    ci_upper = kappa + z * se

    return kappa, ci_lower, ci_upper

def compute_metrics_for_group(
    group: pd.DataFrame,
    clinician_consensus: pd.Series | None = None,
) -> dict:
    """
    Compute all classification metrics for a DataFrame subset.

    If clinician_consensus is provided, Cohen's kappa is computed between
    the clinician consensus and the LLM predictions (inter-rater agreement).
    Otherwise kappa is computed between ground truth and predictions.
    """
    y_true = group["Ground_Truth_Label"].values
    y_pred = group["Predicted_Label"].values
    all_labels = sorted(set(y_true) | set(y_pred))

    n = len(group)
    k = int((y_true == y_pred).sum())
    accuracy = k / n

    ci_lower, ci_upper = proportion_confint(count=k, nobs=n, alpha=0.05, method="wilson")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)

    _, macro_sens, macro_spec, macro_precision, weighted_sens, weighted_spec, weighted_precision = compute_per_class_sens_spec(
        y_true, y_pred, all_labels
    )

    # --- Kappa: clinician consensus vs. LLM predictions ---
    if clinician_consensus is not None:
        # Align on shared vignette IDs that have a consensus
        shared_idx = group.index.intersection(clinician_consensus.index)
        if len(shared_idx) > 0:
            y_clinician = clinician_consensus.loc[shared_idx].values
            y_pred_kappa = group.loc[shared_idx, "Predicted_Label"].values
            kappa, kappa_ci_lower, kappa_ci_upper = compute_kappa_ci(
                y_clinician, y_pred_kappa
            )
            n_kappa = len(shared_idx)
        else:
            kappa, kappa_ci_lower, kappa_ci_upper, n_kappa = np.nan, np.nan, np.nan, 0
    else:
        kappa, kappa_ci_lower, kappa_ci_upper = compute_kappa_ci(y_true, y_pred)
        n_kappa = n

    return {
        "n_vignettes": n,
        "n_correct": k,
        "accuracy": round(accuracy, 4),
        "CI_lower": round(ci_lower, 4),
        "CI_upper": round(ci_upper, 4),
        "CI_Wilson": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        "balanced_accuracy": round(bal_acc, 4),
        "macro_sensitivity": round(macro_sens, 4),
        "macro_specificity": round(macro_spec, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_sensitivity": round(weighted_sens, 4),
        "weighted_specificity": round(weighted_spec, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_f1": round(weighted_f1, 4),
        "kappa": round(kappa, 4),
        "kappa_CI_lower": round(kappa_ci_lower, 4),
        "kappa_CI_upper": round(kappa_ci_upper, 4),
        "kappa_CI": f"[{kappa_ci_lower:.4f}, {kappa_ci_upper:.4f}]",
        "n_kappa_vignettes": n_kappa,
    }


def compute_accuracy_wilson(group: pd.DataFrame) -> dict:
    """
    Compute accuracy + Wilson 95% CI for a group.

    Lighter version of compute_metrics_for_group — used in ablation scripts
    where only accuracy is needed.
    """
    y_true = group["Ground_Truth_Label"].values
    y_pred = group["Predicted_Label"].values

    n = len(group)
    k = int((y_true == y_pred).sum())
    accuracy = k / n

    ci_lower, ci_upper = proportion_confint(count=k, nobs=n, alpha=0.05, method="wilson")

    return {
        "accuracy": round(accuracy, 4),
        "CI_lower": round(ci_lower, 4),
        "CI_upper": round(ci_upper, 4),
        "formatted": f"{accuracy:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]",
    }


def compute_per_class_table(group: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-class sensitivity/specificity/precision table for one model,
    with macro and weighted summary rows appended at the bottom.
    """
    y_true = group["Ground_Truth_Label"].values
    y_pred = group["Predicted_Label"].values
    all_labels = sorted(set(y_true) | set(y_pred))

    (
        per_class_df,
        macro_sens, macro_spec, macro_prec,
        weighted_sens, weighted_spec, weighted_prec,
    ) = compute_per_class_sens_spec(y_true, y_pred, all_labels)

    summary_rows = pd.DataFrame([
        {
            "label": "--- MACRO AVERAGE ---",
            "support": "",
            "sensitivity": round(macro_sens, 4),
            "specificity": round(macro_spec, 4),
            "precision": round(macro_prec, 4),
        },
        {
            "label": "--- WEIGHTED AVERAGE ---",
            "support": "",
            "sensitivity": round(weighted_sens, 4),
            "specificity": round(weighted_spec, 4),
            "precision": round(weighted_prec, 4),
        },
    ])
    return pd.concat([per_class_df, summary_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics aggregation helpers
# ---------------------------------------------------------------------------

def build_metrics_sheets(
    combined: pd.DataFrame,
    clinician_consensus: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    sheets = {}

    overall_records = []
    for llm, group in combined.groupby("llm"):
        record = {"llm": llm}
        record.update(compute_metrics_for_group(group, clinician_consensus))
        overall_records.append(record)
    sheets["Overall"] = reindex_by_model_order(
        pd.DataFrame(overall_records).set_index("llm")
    )

    for cat in sorted(combined["Category"].unique()):
        cat_data = combined[combined["Category"] == cat]
        cat_records = []
        for llm, group in cat_data.groupby("llm"):
            record = {"llm": llm}
            record.update(compute_metrics_for_group(group, clinician_consensus))
            cat_records.append(record)
        sheets[cat] = reindex_by_model_order(
            pd.DataFrame(cat_records).set_index("llm")
        )

    return sheets


def build_misclassification_tables(
    combined: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build misclassification frequency tables (overall + per category).

    For every (True Label, Predicted Label) pair across ALL models pooled,
    count how often each misclassification occurs.  The diagonal (correct
    predictions) is included for completeness.

    Returns a dict of sheet-ready DataFrames:
        "Misclassifications_Overall"   — all vignettes
        "Misclassifications_<Category>" — one per category (Anxiety, Mood, Stress)

    Each table has columns:
        True_Label, Predicted_Label, Count, Pct_of_Total, Correct
    sorted by Count descending, so the most common confusions appear first.
    """

    def _freq_table(df: pd.DataFrame) -> pd.DataFrame:
        freq = (
            df.groupby(["Ground_Truth_Label", "Predicted_Label"])
            .size()
            .reset_index(name="Count")
        )
        freq = freq.sort_values("Count", ascending=False).reset_index(drop=True)
        total = freq["Count"].sum()
        freq["Pct_of_Total"] = (freq["Count"] / total * 100).round(2)
        freq["Correct"] = (
            freq["Ground_Truth_Label"] == freq["Predicted_Label"]
        )
        freq = freq.rename(columns={
            "Ground_Truth_Label": "True_Label",
        })
        return freq

    sheets: dict[str, pd.DataFrame] = {}
    sheets["Misclassifications_Overall"] = _freq_table(combined)

    for cat in sorted(combined["Category"].unique()):
        cat_data = combined[combined["Category"] == cat]
        sheets[f"Misclassifications_{cat}"] = _freq_table(cat_data)

    return sheets


def build_label_distribution(
    combined: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build ground-truth label frequency tables (overall + per category).

    Uses unique vignettes only (deduplicated across models) so counts
    reflect the actual vignette set, not n_models × n_vignettes.

    Returns a dict of sheet-ready DataFrames:
        "Labels_Overall"      — all vignettes
        "Labels_<Category>"   — one per category (Anxiety, Mood, Stress)

    Each table has columns:
        True_Label, Count, Pct_of_Total
    sorted by Count descending.
    """

    # Deduplicate: one row per vignette (drop model duplicates)
    vignettes = combined.drop_duplicates(subset=["Ground_Truth_Label", "Category"], keep="first")
    # In case multiple vignettes share the same label+category, use the index (Vignette_ID)
    vignettes = combined.groupby(combined.index).first()

    def _dist_table(df: pd.DataFrame) -> pd.DataFrame:
        freq = (
            df["Ground_Truth_Label"]
            .value_counts()
            .reset_index()
        )
        freq.columns = ["True_Label", "Count"]
        total = freq["Count"].sum()
        freq["Pct_of_Total"] = (freq["Count"] / total * 100).round(2)
        return freq

    sheets: dict[str, pd.DataFrame] = {}
    sheets["Labels_Overall"] = _dist_table(vignettes)

    for cat in sorted(vignettes["Category"].unique()):
        cat_data = vignettes[vignettes["Category"] == cat]
        sheets[f"Labels_{cat}"] = _dist_table(cat_data)

    return sheets


def build_per_class_sheets(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build one per-class sens/spec/precision sheet per model, ordered by MODEL_ORDER."""
    ordered_models = sort_models(combined["llm"].unique().tolist())
    return {
        f"PerClass_{llm}": compute_per_class_table(combined[combined["llm"] == llm])
        for llm in ordered_models
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_excel(sheets: dict[str, pd.DataFrame], path: Path, index: bool = True) -> None:
    """
    Save a dict of DataFrames to an Excel workbook.

    Sheet names are truncated to 31 chars (Excel limit).
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=index)
    print(f"Saved: {path}")


def print_sheets(sheets: dict[str, pd.DataFrame]) -> None:
    """Print each sheet to stdout."""
    for name, df_sheet in sheets.items():
        print(f"\n{'='*60}\nSheet: {name}\n{'='*60}")
        print(df_sheet.to_string())


# ---------------------------------------------------------------------------
# Top-N accuracy
# ---------------------------------------------------------------------------

def extract_topn_predictions(model_diagnoses_str, n: int) -> list[str]:
    """
    Extract the top-n diagnoses from the Model_Diagnoses column.
    Returns a list of up to n label strings (consolidated via LABEL_MAP).
    """
    try:
        diags = ast.literal_eval(model_diagnoses_str)
        if isinstance(diags, list):
            return [consolidate_label(d) for d in diags[:n]]
    except (ValueError, SyntaxError):
        pass
    raw = str(model_diagnoses_str).strip()
    return [consolidate_label(raw)] if raw else []


def is_correct_topn(row: pd.Series, n: int) -> bool:
    """
    Return True if Ground_Truth_Label appears anywhere in the top-n predictions.
    """
    preds = extract_topn_predictions(row["Model_Diagnoses"], n)
    return row["Ground_Truth_Label"] in preds


def compute_topn_accuracy(group: pd.DataFrame, n: int) -> dict:
    """
    Compute top-n accuracy + Wilson 95% CI for a model group.

    A vignette is 'correct' if the ground truth label appears in
    the model's top-n ranked predictions.
    """
    hits = group.apply(lambda row: is_correct_topn(row, n), axis=1)
    k = int(hits.sum())
    total = len(group)
    accuracy = k / total
    ci_lower, ci_upper = proportion_confint(
        count=k, nobs=total, alpha=0.05, method="wilson"
    )
    return {
        "n_vignettes": total,
        "n_correct": k,
        "accuracy": round(accuracy, 4),
        "CI_lower": round(ci_lower, 4),
        "CI_upper": round(ci_upper, 4),
        "CI_Wilson": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
    }


def build_topn_sheet(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Build a top-n accuracy table (n=1,2,3) for each model across all vignettes.

    Returns a DataFrame with a MultiIndex column structure:
        Model | Top-1 accuracy | Top-1 CI_lower | Top-1 CI_upper | Top-1 CI_Wilson
               | Top-2 ...     | Top-3 ...

    Rows follow MODEL_ORDER.
    """
    records = []
    for llm, group in combined.groupby("llm"):
        row: dict = {"llm": llm}
        for n in (1, 2, 3):
            metrics = compute_topn_accuracy(group, n)
            for key, val in metrics.items():
                row[f"top{n}_{key}"] = val
        records.append(row)

    df = pd.DataFrame(records).set_index("llm")
    return reindex_by_model_order(df)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
LABEL_ABBREVIATIONS = {
    "Post-Traumatic Stress Disorder (PTSD)":         "PTSD",
    "Complex Post-Traumatic Stress Disorder (CPTSD)": "C-PTSD",
    "Prolonged Grief Disorder":                      "PGD",
    "Generalized Anxiety Disorder":                  "GAD",
    "Bipolar Type II Disorder":                      "Bipolar II",
    "Bipolar Type I Disorder":                       "Bipolar I",
    "A diagnosis from a different diagnostic area (e.g. mood disorders, psychotic disorders, personality disorders)" : "Different Diagnosis",
    "A different diagnosis not listed above" : "Different Diagnosis",
    "Other Disorder Specifically Associated with Stress" : "Different Stress-related Diagnosis",
    "Major Depressive Disorder" : "MDD"
}

def _abbreviate(label: str) -> str:
    return LABEL_ABBREVIATIONS.get(label, label)

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    title: str,
    save_path: Path,
    figsize: tuple | None = None,
    fontsize_annot: int = 9,
    fontsize_labels: int = 10,
    max_label_chars: int = 20,
    category: str = None,
    language: str = None,
) -> None:
    """Plot and save a confusion matrix heatmap.

    For English vignettes with a known category, rows (true labels) and
    columns (predicted labels) follow ENGLISH_CM_LABEL_ORDER.  Any predicted
    label not in the canonical list is appended as an extra column only,
    producing an asymmetric matrix when models hallucinate novel diagnoses.
    """
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS',
    ]
    plt.rcParams['axes.unicode_minus'] = False

    # ------------------------------------------------------------------
    # Determine row / column labels
    # ------------------------------------------------------------------
    if (language == "english") and (category in ENGLISH_CM_LABEL_ORDER):
        print(language, category)
        canonical = ENGLISH_CM_LABEL_ORDER[category]

        true_set = set(y_true)
        pred_set = set(y_pred)

        # True-label rows: canonical order, only labels that appear in y_true
        row_labels = [l for l in canonical if l in true_set]

        # Predicted-label cols: canonical order (present in y_pred) + extras
        col_labels = [l for l in canonical if l in pred_set]
        extras = sorted(l for l in pred_set if l not in set(canonical))
        col_labels = col_labels + extras

        # Build asymmetric confusion matrix manually
        label_to_row = {l: i for i, l in enumerate(row_labels)}
        label_to_col = {l: i for i, l in enumerate(col_labels)}
        cm = np.zeros((len(row_labels), len(col_labels)), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            r = label_to_row.get(yt)
            c = label_to_col.get(yp)
            if r is not None and c is not None:
                cm[r, c] += 1

        display_row_labels = row_labels
        display_col_labels = col_labels
    else:
        # Default: symmetric matrix with the labels passed in
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        display_row_labels = labels
        display_col_labels = labels

    display_row_labels = [_abbreviate(l) for l in display_row_labels]
    display_col_labels = [_abbreviate(l) for l in display_col_labels]

    # ------------------------------------------------------------------
    # Wrap long labels
    # ------------------------------------------------------------------
    def _wrap(label, max_chars):
        if len(label) <= max_chars:
            return label
        # Find the space closest to the midpoint
        mid = len(label) // 2
        # Search outward from the midpoint for the nearest space
        best = -1
        for offset in range(mid + 1):
            if mid + offset < len(label) and label[mid + offset] == " ":
                best = mid + offset
                break
            if mid - offset >= 0 and label[mid - offset] == " ":
                best = mid - offset
                break
        if best == -1:
            best = max_chars  # no space found, hard break
        return label[:best] + "\n" + label[best:].strip()

    wrapped_rows = [_wrap(l, max_label_chars) for l in display_row_labels]
    wrapped_cols = [_wrap(l, max_label_chars) for l in display_col_labels]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    
    if figsize is None:
        n_cols = len(display_col_labels)
        n_rows = len(display_row_labels)
        figsize = (max(8, n_cols * 0.6), max(6, n_rows * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=wrapped_cols,
        yticklabels=wrapped_rows,
        ax=ax,
        annot_kws={"size": fontsize_annot},
    )
    ax.set_xlabel("Predicted diagnosis", fontsize=fontsize_labels)
    ax.set_ylabel("True diagnosis", fontsize=fontsize_labels)
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=fontsize_labels)
    plt.yticks(rotation=0, fontsize=fontsize_labels)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

def plot_confusion_matrix_pct(
    y_true,
    y_pred,
    labels: list[str],
    title: str,
    save_path: Path,
    figsize: tuple | None = None,
    fontsize_annot: int = 9,
    fontsize_labels: int = 10,
    max_label_chars: int = 20,
    category: str = None,
    language: str = None,
) -> None:
    """Plot and save a row-normalized confusion matrix heatmap (percentages).

    Each cell shows the fraction of true-label instances predicted as the
    column label, so each row sums to 1.0.  Annotation format is e.g. '0.83'.

    For English vignettes with a known category, the same canonical label
    ordering and asymmetric-matrix logic as plot_confusion_matrix() applies.
    """
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS',
    ]
    plt.rcParams['axes.unicode_minus'] = False

    # ------------------------------------------------------------------
    # Build raw count matrix (same logic as plot_confusion_matrix)
    # ------------------------------------------------------------------
    if (language == "english") and (category in ENGLISH_CM_LABEL_ORDER):
        canonical = ENGLISH_CM_LABEL_ORDER[category]

        true_set = set(y_true)
        pred_set = set(y_pred)

        row_labels = [l for l in canonical if l in true_set]
        col_labels = [l for l in canonical if l in pred_set]
        extras = sorted(l for l in pred_set if l not in set(canonical))
        col_labels = col_labels + extras

        label_to_row = {l: i for i, l in enumerate(row_labels)}
        label_to_col = {l: i for i, l in enumerate(col_labels)}
        cm_counts = np.zeros((len(row_labels), len(col_labels)), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            r = label_to_row.get(yt)
            c = label_to_col.get(yp)
            if r is not None and c is not None:
                cm_counts[r, c] += 1

        display_row_labels = row_labels
        display_col_labels = col_labels
    else:
        cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
        display_row_labels = labels
        display_col_labels = labels

    display_row_labels = [_abbreviate(l) for l in display_row_labels]
    display_col_labels = [_abbreviate(l) for l in display_col_labels]

    # ------------------------------------------------------------------
    # Row-normalize: divide each row by its sum
    # Rows with zero true instances become NaN to avoid division by zero
    # ------------------------------------------------------------------
    row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore"):
        cm_pct = np.where(row_sums > 0, cm_counts / row_sums, np.nan)

    # ------------------------------------------------------------------
    # Wrap long labels (identical helper as in plot_confusion_matrix)
    # ------------------------------------------------------------------
    def _wrap(label, max_chars):
        if len(label) <= max_chars:
            return label
        mid = len(label) // 2
        best = -1
        for offset in range(mid + 1):
            if mid + offset < len(label) and label[mid + offset] == " ":
                best = mid + offset
                break
            if mid - offset >= 0 and label[mid - offset] == " ":
                best = mid - offset
                break
        if best == -1:
            best = max_chars
        return label[:best] + "\n" + label[best:].strip()

    wrapped_rows = [_wrap(l, max_label_chars) for l in display_row_labels]
    wrapped_cols = [_wrap(l, max_label_chars) for l in display_col_labels]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if figsize is None:
        n_cols = len(display_col_labels)
        n_rows = len(display_row_labels)
        figsize = (max(8, n_cols * 0.6), max(6, n_rows * 0.5))

    annot_array = np.empty(cm_pct.shape, dtype=object)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            if np.isnan(val):
                annot_array[i, j] = ""
            else:
                annot_array[i, j] = f"{int(round(val * 100))}"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_pct,
        annot=annot_array,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=wrapped_cols,
        yticklabels=wrapped_rows,
        ax=ax,
        annot_kws={"size": fontsize_annot},
    )
    ax.set_xlabel("Predicted diagnosis", fontsize=fontsize_labels)
    ax.set_ylabel("True diagnosis", fontsize=fontsize_labels)
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=fontsize_labels)
    plt.yticks(rotation=0, fontsize=fontsize_labels)

    for text in ax.texts:
        try:
            val = float(text.get_text())
        except ValueError:
            continue
        text.set_color("white" if val >= 60 else "#444444")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

def plot_confusion_matrix_pct_fixed(
    y_true,
    y_pred,
    labels: list[str],
    title: str,
    save_path: Path,
    figsize: tuple | None = None,
    fontsize_labels: int = 14,
    max_label_chars: int = 20,
    cell_size: float = 0.55,
    category: str = None,
    language: str = None,
) -> None:
    """Plot and save a row-normalized confusion matrix heatmap (percentages).

    Each cell shows the fraction of true-label instances predicted as the
    column label, so each row sums to 1.0.  Annotation format is e.g. '83'.

    Cell size is fixed via `cell_size` (inches per cell) so squares stay
    uniform regardless of how many labels are present.  Annotation font size
    scales automatically with cell_size.

    For English vignettes with a known category, the same canonical label
    ordering and asymmetric-matrix logic as plot_confusion_matrix() applies.
    """
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS',
    ]
    plt.rcParams['axes.unicode_minus'] = False

    # ------------------------------------------------------------------
    # Build raw count matrix (same logic as plot_confusion_matrix)
    # ------------------------------------------------------------------
    if (language == "english") and (category in ENGLISH_CM_LABEL_ORDER):
        canonical = ENGLISH_CM_LABEL_ORDER[category]

        true_set = set(y_true)
        pred_set = set(y_pred)

        row_labels = [l for l in canonical if l in true_set]
        col_labels = [l for l in canonical if l in pred_set]
        extras = sorted(l for l in pred_set if l not in set(canonical))
        col_labels = col_labels + extras

        label_to_row = {l: i for i, l in enumerate(row_labels)}
        label_to_col = {l: i for i, l in enumerate(col_labels)}
        cm_counts = np.zeros((len(row_labels), len(col_labels)), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            r = label_to_row.get(yt)
            c = label_to_col.get(yp)
            if r is not None and c is not None:
                cm_counts[r, c] += 1

        display_row_labels = row_labels
        display_col_labels = col_labels
    else:
        cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
        display_row_labels = labels
        display_col_labels = labels

    display_row_labels = [_abbreviate(l) for l in display_row_labels]
    display_col_labels = [_abbreviate(l) for l in display_col_labels]

    x_map = CM_X_AXIS_MAP[category]
    y_map = CM_Y_AXIS_MAP[category]
    display_col_labels = [x_map.get(l, _abbreviate(l)) for l in display_col_labels]
    display_row_labels = [y_map.get(l, _abbreviate(l)) for l in display_row_labels]


    # ------------------------------------------------------------------
    # Row-normalize: divide each row by its sum
    # Rows with zero true instances become NaN to avoid division by zero
    # ------------------------------------------------------------------
    row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore"):
        cm_pct = np.where(row_sums > 0, cm_counts / row_sums, np.nan)

    # ------------------------------------------------------------------
    # Wrap long labels
    # ------------------------------------------------------------------
    def _wrap(label, max_chars):
        if len(label) <= max_chars:
            return label
        mid = len(label) // 2
        best = -1
        for offset in range(mid + 1):
            if mid + offset < len(label) and label[mid + offset] == " ":
                best = mid + offset
                break
            if mid - offset >= 0 and label[mid - offset] == " ":
                best = mid - offset
                break
        if best == -1:
            best = max_chars
        return label[:best] + "\n" + label[best:].strip()

    wrapped_rows = [_wrap(l, max_label_chars) for l in display_row_labels]
    wrapped_cols = [_wrap(l, max_label_chars) for l in display_col_labels]

    # ------------------------------------------------------------------
    # Compute figsize from fixed cell size so squares stay uniform
    # ------------------------------------------------------------------
    if figsize is None:
        n_cols = len(display_col_labels)
        n_rows = len(display_row_labels)
        margin_w = 3.5   # space for y-axis labels + colorbar
        margin_h = 2.5   # space for x-axis labels + title
        figsize = (n_cols * cell_size + margin_w, n_rows * cell_size + margin_h)

    # Annotation font size scales with cell size
    fontsize_annot = max(7, int(cell_size * 22))  # bigger text relative to box

    # ------------------------------------------------------------------
    # Build annotation array
    # ------------------------------------------------------------------
    annot_array = np.empty(cm_pct.shape, dtype=object)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            annot_array[i, j] = "" if np.isnan(val) else f"{int(round(val * 100))}"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_pct,
        annot=annot_array,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=wrapped_cols,
        yticklabels=wrapped_rows,
        ax=ax,
        annot_kws={"size": fontsize_annot},
        cbar=False
    )
    # ax.set_xlabel("Predicted diagnosis", fontsize=fontsize_labels)
    # ax.set_ylabel("True diagnosis", fontsize=fontsize_labels)
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=0, fontsize=fontsize_labels)
    plt.yticks(rotation=0, fontsize=fontsize_labels)

    for text in ax.texts:
        try:
            val = float(text.get_text())
        except ValueError:
            continue
        text.set_color("white" if val >= 60 else "#444444")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")