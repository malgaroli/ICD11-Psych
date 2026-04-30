"""
run_clinician_metrics.py
=========================
Compute classification metrics for *clinician* diagnostic vignette ratings,
mirroring the LLM metrics produced by run_classification_metrics.py.

All multilingual labels are mapped to their English canonical form before
metric computation, so weighted metrics are computed over a unified label set.

Subtypes of Bipolar I, Bipolar II, Recurrent Depressive Disorder, and
Single Episode Depressive Disorder are collapsed to their parent category
(but comorbid "AND" labels are preserved).

Input:
    clinicians_cleaned.xlsx with columns:
        ID, Category, Vignette_ID, Ground_Truth, Final_Answer

Outputs (saved to results_Apr26/_results/clinicians/):
    - clinician_classification_metrics.xlsx
    - clinician_per_rater_accuracy.xlsx
    - clinician_vignette_weighted_accuracy.xlsx
    - Confusion matrix PNGs (overall + per category)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import json

from icd11_utils import (
    compute_metrics_for_group,
    compute_per_class_table,
    compute_accuracy_wilson,
    plot_confusion_matrix,
    save_excel,
    print_sheets,
    plot_confusion_matrix_pct,
    plot_confusion_matrix_pct_fixed
)
import matplotlib.colors as mcolors

# custom_cmap_cli = mcolors.LinearSegmentedColormap.from_list(
#     "custom_blue", ["#FFFFFF", "#B2CDE7", "#4A7FAD"]
# )
# custom_cmap_cli = mcolors.LinearSegmentedColormap.from_list(
#     "custom_blue", custom_cmap_cli(np.linspace(0, 1, 256)**0.5)
# )
custom_cmap_cli = mcolors.LinearSegmentedColormap.from_list(
    "custom_blue", ["#FFFFFF", "#B2CDE7", "#2E5F8A"]  # darker endpoint
)
custom_cmap_cli = mcolors.LinearSegmentedColormap.from_list(
    "custom_blue", custom_cmap_cli(np.linspace(0, 1, 256)**0.5)  # stronger gamma
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
RESULTS_FOLDER = BASE_PATH / "results_resubmission"
CLINICIAN_FILE = RESULTS_FOLDER / "clinicians" / "cleaned" / "clinicians_cleaned.xlsx"

OUTPUT_DIR = RESULTS_FOLDER / "_results" / "clinicians"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Multilingual → English label mapping
# ---------------------------------------------------------------------------

MULTILINGUAL_TO_ENGLISH = {
    # ------------------------------------------------------------------
    # No Diagnosis — all languages
    # ------------------------------------------------------------------
    "no diagnosis":       "No Diagnosis",
    "No Diagnosis":       "No Diagnosis",
    "Sin diagnóstico":    "No Diagnosis",
    "ningún diagnóstico": "No Diagnosis",
    "Pas de diagnostic":  "No Diagnosis",
    "Отсутствие диагноза":"No Diagnosis",
    "診断なし":            "No Diagnosis",
    "不予诊断":            "No Diagnosis",
    "nan":                "No Diagnosis",

    # ------------------------------------------------------------------
    # A different diagnosis — all languages
    # ------------------------------------------------------------------
    "a different diagnosis":   "A different diagnosis",
    "A different diagnosis":   "A different diagnosis",
    "Un diagnóstico diferente":"A different diagnosis",
    "un diagnóstico diferente":"A different diagnosis",
    "Un diagnostic different": "A different diagnosis",
    "Un diagnostic différent": "A different diagnosis",
    "Другой диагноз":         "A different diagnosis",
    "別の診断":                "A different diagnosis",
    "別の診断（ここには挙げられていない診断）": "A different diagnosis",
    "其他诊断":                "A different diagnosis",

    # ------------------------------------------------------------------
    # Stress category (A-series) — harmonise to canonical CM labels
    # ------------------------------------------------------------------
    "Post-Traumatic Stress Disorder":         "Post-Traumatic Stress Disorder (PTSD)",
    "Complex Post-Traumatic Stress Disorder":  "Complex Post-Traumatic Stress Disorder (CPTSD)",

    # ------------------------------------------------------------------
    # P-series (Stress)
    # ------------------------------------------------------------------
    "Trastorno de Duelo Prolongado":  "Prolonged Grief Disorder",
    "遷延性悲嘆障害":                  "Prolonged Grief Disorder",
    "Затяжная реакция горя":          "Prolonged Grief Disorder",

    "Trastorno de Adaptación":        "Adjustment Disorder",
    "適応障害":                        "Adjustment Disorder",
    "Расстройство адаптации":         "Adjustment Disorder",
    "Расстройство адаптаци":          "Adjustment Disorder",

    # ------------------------------------------------------------------
    # N-series (Anxiety: standalone GAD)
    # ------------------------------------------------------------------
    "Trastorno de Ansiedad Generalizada":      "Generalized Anxiety Disorder",
    "Генерализованное тревожное расстройство": "Generalized Anxiety Disorder",
    "全般性不安障害":                            "Generalized Anxiety Disorder",

    # ------------------------------------------------------------------
    # Q-series (Anxiety)
    # ------------------------------------------------------------------
    # Q1 GAD
    "Trastorno de ansiedad generalizada":      "Generalized Anxiety Disorder",
    "Trouble d'anxiété généralisée":           "Generalized Anxiety Disorder",
    "广泛性焦虑障碍":                            "Generalized Anxiety Disorder",

    # Q1 + Q2 combo
    "Generalized Anxiety Disorder AND Q2 Panic Disorder":
        "Generalized Anxiety Disorder AND Panic Disorder",
    "Trastorno de ansiedad generalizada Y Q2 Trastorno de pánico":
        "Generalized Anxiety Disorder AND Panic Disorder",
    "Trouble d'anxiété généralisée ET Q2 Trouble panique":
        "Generalized Anxiety Disorder AND Panic Disorder",
    "Генерализованное тревожное расстройство И Q2 Паническое расстройство":
        "Generalized Anxiety Disorder AND Panic Disorder",
    "广泛性焦虑障碍 并 Q2 惊恐障碍":
        "Generalized Anxiety Disorder AND Panic Disorder",

    # Q2 Panic Disorder
    "Trastorno de pánico":       "Panic Disorder",
    "Trouble panique":           "Panic Disorder",
    "Паническое расстройство":   "Panic Disorder",
    "パニック障害":               "Panic Disorder",
    "惊恐障碍":                   "Panic Disorder",

    # Q3 Agoraphobia
    "Agorafobia":                "Agoraphobia",
    "Agoraphobie":               "Agoraphobia",
    "Агорафобия":                "Agoraphobia",
    "広場恐怖症":                 "Agoraphobia",
    "场所恐惧症":                 "Agoraphobia",

    # Q3 + Q2 combo
    "Agoraphobia AND Q2 Panic Disorder":
        "Agoraphobia AND Panic Disorder",
    "Agorafobia Y Q2 Trastorno de pánico":
        "Agoraphobia AND Panic Disorder",
    "Agoraphobie ET Q2 Trouble panique":
        "Agoraphobia AND Panic Disorder",
    "Агорафобия И Q2 Паническое расстройство":
        "Agoraphobia AND Panic Disorder",
    "広場恐怖症 および Q2 パニック障害":
        "Agoraphobia AND Panic Disorder",
    "场所恐惧症 并 Q2 惊恐障碍":
        "Agoraphobia AND Panic Disorder",

    # Q4 Specific Phobia
    "Fobia específica":          "Specific Phobia",
    "Phobie spécifique":         "Specific Phobia",
    "Специфическая фобия":       "Specific Phobia",
    "限局性恐怖症":               "Specific Phobia",
    "特定恐惧症":                 "Specific Phobia",

    # Q4 + Q2 combo
    "Specific Phobia AND Q2 Panic Disorder":
        "Specific Phobia AND Panic Disorder",
    "Fobia específica Y Q2 Trastorno de pánico":
        "Specific Phobia AND Panic Disorder",
    "特定恐惧症 并 Q2 惊恐障碍":
        "Specific Phobia AND Panic Disorder",

    # Q5 Social Anxiety Disorder
    "Trastorno de ansiedad social":       "Social Anxiety Disorder",
    "Trouble d'anxiété sociale":          "Social Anxiety Disorder",
    "Социальное тревожное расстройство":  "Social Anxiety Disorder",
    "社交不安障害":                        "Social Anxiety Disorder",
    "社交焦虑障碍":                        "Social Anxiety Disorder",

    # Q5 + Q2 combo
    "Social Anxiety Disorder AND Q2 Panic Disorder":
        "Social Anxiety Disorder AND Panic Disorder",
    "Trastorno de ansiedad social Y Q2 Trastorno de pánico":
        "Social Anxiety Disorder AND Panic Disorder",
    "Trouble d'anxiété sociale ET Q2 Trouble panique":
        "Social Anxiety Disorder AND Panic Disorder",
    "Социальное тревожное расстройство И Q2 Паническое расстройство":
        "Social Anxiety Disorder AND Panic Disorder",
    "社交不安障害 および Q2 パニック障害":
        "Social Anxiety Disorder AND Panic Disorder",
    "社交焦虑障碍 并 Q2 惊恐障碍":
        "Social Anxiety Disorder AND Panic Disorder",

    # Q6 Separation Anxiety Disorder
    "Trastorno de ansiedad por separación":       "Separation Anxiety Disorder",
    "Trouble d'anxiété de séparation":            "Separation Anxiety Disorder",
    "Сепарационное тревожное расстройство":       "Separation Anxiety Disorder",
    "分離不安障害":                                "Separation Anxiety Disorder",
    "分离性焦虑障碍":                              "Separation Anxiety Disorder",

    # Q6 + Q2 combo
    "Separation Anxiety Disorder AND Q2 Panic Disorder":
        "Separation Anxiety Disorder AND Panic Disorder",
    "Trastorno de ansiedad por separación Y Q2 Trastorno de pánico":
        "Separation Anxiety Disorder AND Panic Disorder",
    "Trouble d'anxiété de séparation ET Q2 Trouble panique":
        "Separation Anxiety Disorder AND Panic Disorder",
    "Сепарационное тревожное расстройство и Q2 Паническое расстройство":
        "Separation Anxiety Disorder AND Panic Disorder",
    "分离性焦虑障碍 并 Q2 惊恐障碍":
        "Separation Anxiety Disorder AND Panic Disorder",

    # Q7 Selective Mutism
    "Mutismo selectivo":         "Selective Mutism",
    "Селективный мутизм":       "Selective Mutism",
    "选择性缄默症":              "Selective Mutism",

    # Q8 Other Anxiety and Fear-Related Disorder
    "Otro trastorno de ansiedad o relacionado con el miedo":
        "Other Anxiety and Fear-Related Disorder",
    "Autre trouble anxieux et lié à la peur":
        "Other Anxiety and Fear-Related Disorder",
    "Другое тревожное и связанное со страхом расстройство":
        "Other Anxiety and Fear-Related Disorder",
    "他の不安と恐怖に関連する障害":
        "Other Anxiety and Fear-Related Disorder",
    "其他焦虑及恐惧相关障碍":
        "Other Anxiety and Fear-Related Disorder",

    # Q9 Unspecified Anxiety and Fear-Related Disorder
    "Trastorno de ansiedad o relacionado con el miedo No especificado":
        "Unspecified Anxiety and Fear-Related Disorder",
    "Trouble anxieux et lié à la peur, sans précision":
        "Unspecified Anxiety and Fear-Related Disorder",
    "未特定的焦虑及恐惧相关障碍":
        "Unspecified Anxiety and Fear-Related Disorder",
    "特定不能の不安と恐怖に関連する障害":
        "Unspecified Anxiety and Fear-Related Disorder",

    # ------------------------------------------------------------------
    # L-series (Mood) — mapped to English subtypes first,
    # then collapse_subtypes() rolls them up to parent categories
    # ------------------------------------------------------------------
    # L1A
    "Episodio Único de Trastorno Depresivo, Leve":
        "Single Episode Depressive Disorder, Mild",
    "Единичный эпизод депрессивного расстройства, легкий":
        "Single Episode Depressive Disorder, Mild",
    "単一エピソード抑うつ障害、軽症":
        "Single Episode Depressive Disorder, Mild",

    # L1B
    "Episodio Único de Trastorno Depresivo, Moderado, sin síntomas psicóticos":
        "Single Episode Depressive Disorder, Moderate, without psychotic symptoms",
    "Единичный эпизод депрессивного расстройства, умеренный, без психотических симптомов":
        "Single Episode Depressive Disorder, Moderate, without psychotic symptoms",
    "単一エピソード抑うつ障害、中等症、精神病症状を伴わない":
        "Single Episode Depressive Disorder, Moderate, without psychotic symptoms",

    # L1C
    "Episodio Único de Trastorno Depresivo, Moderado, con síntomas psicóticos":
        "Single Episode Depressive Disorder, Moderate, with psychotic symptoms",
    "Единичный эпизод депрессивного расстройства, умеренный, с психотическими симптомами":
        "Single Episode Depressive Disorder, Moderate, with psychotic symptoms",
    "単一エピソード抑うつ障害、中等症、精神病症状を伴う":
        "Single Episode Depressive Disorder, Moderate, with psychotic symptoms",

    # L1D
    "Episodio Único de Trastorno Depresivo, Grave, sin síntomas psicóticos":
        "Single Episode Depressive Disorder, Severe, without psychotic symptoms",
    "Единичный эпизод депрессивного расстройства, тяжелый, без психотических симптомов":
        "Single Episode Depressive Disorder, Severe, without psychotic symptoms",
    "単一エピソード抑うつ障害、重症、精神病症状を伴わない":
        "Single Episode Depressive Disorder, Severe, without psychotic symptoms",

    # L1E
    "Episodio Único de Trastorno Depresivo, Grave, con síntomas psicóticos":
        "Single Episode Depressive Disorder, Severe, with psychotic symptoms",
    "Единичный эпизод депрессивного расстройства, тяжелый, с психотическими симптомами":
        "Single Episode Depressive Disorder, Severe, with psychotic symptoms",
    "単一エピソード抑うつ障害、重症、精神病症状を伴う":
        "Single Episode Depressive Disorder, Severe, with psychotic symptoms",

    # L2A
    "Trastorno Depresivo Recurrente, Episodio Actual Leve":
        "Recurrent Depressive Disorder, Current Episode Mild",
    "Рекуррентное депрессивное расстройство, текущий эпизод легкий":
        "Recurrent Depressive Disorder, Current Episode Mild",
    "反復性抑うつ障害、現在のエピソードは軽症":
        "Recurrent Depressive Disorder, Current Episode Mild",

    # L2B
    "Trastorno Depresivo Recurrente, Episodio Actual Moderado, sin síntomas psicóticos":
        "Recurrent Depressive Disorder, Current Episode Moderate, without psychotic symptoms",
    "Рекуррентное депрессивное расстройство, текущий эпизод умеренный, без психотических симптомов":
        "Recurrent Depressive Disorder, Current Episode Moderate, without psychotic symptoms",
    "反復性抑うつ障害、現在のエピソードは中等症、精神病症状を伴わない":
        "Recurrent Depressive Disorder, Current Episode Moderate, without psychotic symptoms",

    # L2C
    "Trastorno Depresivo Recurrente, Episodio Actual Moderado, con síntomas psicóticos":
        "Recurrent Depressive Disorder, Current Episode Moderate, with psychotic symptoms",
    "Рекуррентное депрессивное расстройство, текущий эпизод умеренный, с психотическими симптомами":
        "Recurrent Depressive Disorder, Current Episode Moderate, with psychotic symptoms",
    "反復性抑うつ障害、現在のエピソードは中等症、精神病症状を伴う":
        "Recurrent Depressive Disorder, Current Episode Moderate, with psychotic symptoms",

    # L2D
    "Trastorno Depresivo Recurrente, Episodio Actual Grave, sin síntomas psicóticos":
        "Recurrent Depressive Disorder, Current Episode Severe, without psychotic symptoms",
    "Рекуррентное депрессивное расстройство, текущий эпизод тяжелый, без психотических симптомов":
        "Recurrent Depressive Disorder, Current Episode Severe, without psychotic symptoms",
    "反復性抑うつ障害、現在のエピソードは重症、精神病症状を伴わない":
        "Recurrent Depressive Disorder, Current Episode Severe, without psychotic symptoms",

    # L2E
    "Trastorno Depresivo Recurrente, Episodio Actual Grave, con síntomas psicóticos":
        "Recurrent Depressive Disorder, Current Episode Severe, with psychotic symptoms",
    "Рекуррентное депрессивное расстройство, текущий эпизод тяжелый, с психотическими симптомами":
        "Recurrent Depressive Disorder, Current Episode Severe, with psychotic symptoms",
    "反復性抑うつ障害、現在のエピソードは重症、精神病症状を伴う":
        "Recurrent Depressive Disorder, Current Episode Severe, with psychotic symptoms",

    # L2F
    "Рекуррентное депрессивное расстройство, текущая неполная  ремиссия":
        "Recurrent Depressive Disorder, currently in partial remission",

    # L3
    "Trastorno Distímico":              "Dysthymic Disorder",
    "Дистимическое расстройство":       "Dysthymic Disorder",
    "気分変調性障害":                     "Dysthymic Disorder",

    # L4
    "Trastorno Mixto Depresivo y de Ansiedad":
        "Mixed Depressive and Anxiety Disorder",
    "Смешанное депрессивное и тревожное расстройство":
        "Mixed Depressive and Anxiety Disorder",
    "混合性抑うつ不安障害":
        "Mixed Depressive and Anxiety Disorder",

    # L5A–L5L (Bipolar Type I)
    "Trastorno Bipolar Tipo I, Episodio Maníaco Actual, sin síntomas psicóticos":
        "Bipolar Type I Disorder, Current Episode Manic, without psychotic symptoms",
    "Биполярное расстройство, I типа, текущий маниакальный эпизод, без психотических симптомов":
        "Bipolar Type I Disorder, Current Episode Manic, without psychotic symptoms",
    "双極I型障害、現在躁病エピソード、精神病症状を伴わない":
        "Bipolar Type I Disorder, Current Episode Manic, without psychotic symptoms",
    "Trastorno Bipolar Tipo I, Episodio Maníaco Actual, con síntomas psicóticos":
        "Bipolar Type I Disorder, Current Episode Manic, with psychotic symptoms",
    "Биполярное расстройство, I типа, текущий маниакальный эпизод, с психотическими симптомами":
        "Bipolar Type I Disorder, Current Episode Manic, with psychotic symptoms",
    "双極I型障害、現在躁病エピソード、精神病症状を伴う":
        "Bipolar Type I Disorder, Current Episode Manic, with psychotic symptoms",
    "Trastorno Bipolar Tipo I, Episodio Hipomaníaco Actual":
        "Bipolar Type I Disorder, Current Episode Hypomanic",
    "双極I型障害、現在軽躁病エピソード":
        "Bipolar Type I Disorder, Current Episode Hypomanic",
    "Trastorno Bipolar Tipo I, Episodio Depresivo Actual, Leve":
        "Bipolar Type I Disorder, Current Episode Depressive, Mild",
    "Биполярное расстройство, I типа, текущий депрессивный эпизод, легкий":
        "Bipolar Type I Disorder, Current Episode Depressive, Mild",
    "双極I型障害、現在抑うつエピソード、軽症":
        "Bipolar Type I Disorder, Current Episode Depressive, Mild",
    "Trastorno Bipolar Tipo I, Episodio Depresivo Actual, Moderado, sin síntomas psicóticos":
        "Bipolar Type I Disorder, Current Episode Depressive, Moderate, without psychotic symptoms",
    "Биполярное расстройство, I типа, текущий депрессивный эпизод, умеренный, без психотических симптомов":
        "Bipolar Type I Disorder, Current Episode Depressive, Moderate, without psychotic symptoms",
    "双極I型障害、現在抑うつエピソード、中等症、精神病症状を伴わない":
        "Bipolar Type I Disorder, Current Episode Depressive, Moderate, without psychotic symptoms",
    "Биполярное расстройство, I типа, текущий депрессивный эпизод, умеренный, с психотическими симптомами":
        "Bipolar Type I Disorder, Current Episode Depressive, Moderate, with psychotic symptoms",
    "双極I型障害、現在抑うつエピソード、中等症、精神病症状を伴う":
        "Bipolar Type I Disorder, Current Episode Depressive, Moderate, with psychotic symptoms",
    "Trastorno Bipolar Tipo I, Episodio Depresivo Actual, Grave, sin síntomas psicóticos":
        "Bipolar Type I Disorder, Current Episode Depressive, Severe, without psychotic symptoms",
    "Биполярное расстройство, I типа, текущий депрессивный эпизод, тяжелый, без психотических симптомов":
        "Bipolar Type I Disorder, Current Episode Depressive, Severe, without psychotic symptoms",
    "Trastorno Bipolar Tipo I, Episodio Depresivo Actual, Grave, con síntomas psicóticos":
        "Bipolar Type I Disorder, Current Episode Depressive, Severe, with psychotic symptoms",
    "Trastorno Bipolar Tipo I, Episodio Mixto Actual, sin síntomas psicóticos":
        "Bipolar Type I Disorder, Current Episode Mixed, without psychotic symptoms",
    "Биполярное расстройство, I типа, текущий смешанный эпизод, без психотических симптомов":
        "Bipolar Type I Disorder, Current Episode Mixed, without psychotic symptoms",
    "双極I型障害、現在混合性エピソード、精神病症状を伴わない":
        "Bipolar Type I Disorder, Current Episode Mixed, without psychotic symptoms",
    "Биполярное расстройство, I типа, текущий смешанный эпизод, с психотическими симптомами":
        "Bipolar Type I Disorder, Current Episode Mixed, with psychotic symptoms",
    "双極I型障害、現在部分寛解、直近のエピソードは躁病または軽躁病":
        "Bipolar Type I Disorder, currently in partial remission, most recent episode manic or hypomanic",
    "Trastorno Bipolar Tipo I, actualmente en remisión parcial, episodio más reciente depresivo":
        "Bipolar Type I Disorder, currently in partial remission, most recent episode depressive",

    # L6A–L6I (Bipolar Type II)
    "Trastorno Bipolar Tipo II, Episodio Hipomaníaco Actual":
        "Bipolar Type II Disorder, Current Episode Hypomanic",
    "Биполярное расстройство II типа, текущий гипоманиакальный эпизод":
        "Bipolar Type II Disorder, Current Episode Hypomanic",
    "双極II型障害、現在軽躁病エピソード":
        "Bipolar Type II Disorder, Current Episode Hypomanic",
    "Trastorno Bipolar Tipo II, Episodio Depresivo Actual, Leve":
        "Bipolar Type II Disorder, Current Episode Depressive, Mild",
    "Биполярное расстройство II типа, текущий депрессивный эпизод, легкий":
        "Bipolar Type II Disorder, Current Episode Depressive, Mild",
    "双極II型障害、現在抑うつエピソード、軽症":
        "Bipolar Type II Disorder, Current Episode Depressive, Mild",
    "Trastorno Bipolar Tipo II, Episodio Depresivo Actual, Moderado sin síntomas psicóticos":
        "Bipolar Type II Disorder, Current Episode Depressive, Moderate, without psychotic symptoms",
    "Биполярное расстройство II типа, текущий депрессивный эпизод, умеренный, без психотических симптомов":
        "Bipolar Type II Disorder, Current Episode Depressive, Moderate, without psychotic symptoms",
    "双極II型障害、現在抑うつエピソード、中等症、精神病症状を伴わない":
        "Bipolar Type II Disorder, Current Episode Depressive, Moderate, without psychotic symptoms",
    "双極II型障害、現在抑うつエピソード、中等症、精神病症状を伴う":
        "Bipolar Type II Disorder, Current Episode Depressive, Moderate, with psychotic symptoms",
    "Trastorno Bipolar Tipo II, Episodio Depresivo Actual, Grave sin síntomas psicóticos":
        "Bipolar Type II Disorder, Current Episode Depressive, Severe, without psychotic symptoms",
    "Биполярное расстройство II типа, текущий депрессивный эпизод, тяжелый, без психотических симптомов":
        "Bipolar Type II Disorder, Current Episode Depressive, Severe, without psychotic symptoms",
    "双極II型障害、現在抑うつエピソード、重症、精神病症状を伴わない":
        "Bipolar Type II Disorder, Current Episode Depressive, Severe, without psychotic symptoms",
    "双極II型障害、現在部分寛解、直近のエピソードは軽躁病":
        "Bipolar Type II Disorder, currently in partial remission, most recent episode hypomanic",
    "Биполярное расстройство II типа, текущая неполная ремиссия, предшествующий эпизод депрессивный":
        "Bipolar Type II Disorder, currently in partial remission, most recent episode depressive",
    "双極II型障害、現在部分寛解、直近のエピソードは抑うつ":
        "Bipolar Type II Disorder, currently in partial remission, most recent episode depressive",

    # L7
    "Trastorno Ciclotímico":           "Cyclothymic Disorder",
    "Циклотимическое расстройство":    "Cyclothymic Disorder",
    "気分循環性障害":                    "Cyclothymic Disorder",

    # L8
    "Otros Trastorno del Estado de Ánimo": "Other Mood Disorder",
    "Другое аффективное расстройство":     "Other Mood Disorder",
    "他の気分障害":                         "Other Mood Disorder",
}


# ---------------------------------------------------------------------------
# Subtype collapsing
# ---------------------------------------------------------------------------

COLLAPSE_PREFIXES = [
    "Bipolar Type I Disorder,",
    "Bipolar Type II Disorder,",
    "Recurrent Depressive Disorder,",
    "Single Episode Depressive Disorder,",
]

COLLAPSE_TARGETS = {
    "Bipolar Type I Disorder,":           "Bipolar Type I Disorder",
    "Bipolar Type II Disorder,":          "Bipolar Type II Disorder",
    "Recurrent Depressive Disorder,":     "Recurrent Depressive Disorder",
    "Single Episode Depressive Disorder,":"Single Episode Depressive Disorder",
}


def collapse_subtypes(label: str) -> str:
    """
    Collapse detailed subtypes to parent category.
    Labels containing " AND " are preserved as-is (comorbid diagnoses).
    """
    if " AND " in label:
        return label
    for prefix in COLLAPSE_PREFIXES:
        if label.startswith(prefix):
            return COLLAPSE_TARGETS[prefix]
    return label


# ---------------------------------------------------------------------------
# Apostrophe / quote normalisation
# ---------------------------------------------------------------------------

def normalise_quotes(s: str) -> str:
    """
    Replace all curly / smart quote variants with straight ASCII equivalents.
    Excel commonly converts ' to \u2019 (right single quotation mark).
    """
    return (
        s.replace("\u2018", "'")   # left single quotation mark
         .replace("\u2019", "'")   # right single quotation mark
         .replace("\u201C", '"')   # left double quotation mark
         .replace("\u201D", '"')   # right double quotation mark
         .replace("\u02BC", "'")   # modifier letter apostrophe
    )


# ---------------------------------------------------------------------------
# map_to_english
# ---------------------------------------------------------------------------

def map_to_english(label: str) -> str:
    """
    Map a (possibly non-English) diagnostic label to its English canonical form,
    then collapse subtypes.

    Steps:
        1. Normalise smart quotes / curly apostrophes to ASCII.
        2. Strip the vignette-code prefix (e.g. "Q1 ", "L5A ", "Р2 ").
        3. Look up in MULTILINGUAL_TO_ENGLISH.
        4. If not found, return the stripped label as-is (assumed English).
        5. Collapse subtypes to parent category.
        6. Harmonise "no diagnosis" variants to "No Diagnosis".
    """
    label = normalise_quotes(str(label).strip())

    # Strip leading vignette code: ASCII or Cyrillic letter(s) + digit(s) + optional letter + space
    stripped = re.sub(r"^[A-Za-zА-Яа-я]\d+[A-Za-z]?\s+", "", label).strip()

    # Try mapping the stripped version first, then the original
    mapped = MULTILINGUAL_TO_ENGLISH.get(stripped)
    if mapped is None:
        mapped = MULTILINGUAL_TO_ENGLISH.get(label)
    if mapped is None:
        if stripped.lower() in ("no diagnosis", "sin diagnóstico", "ningún diagnóstico",
                                "pas de diagnostic", "nan"):
            mapped = "No Diagnosis"
        else:
            mapped = stripped

    mapped = collapse_subtypes(mapped)
    return mapped


# ---------------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------------
def load_clinician_ratings(path: Path) -> pd.DataFrame:
    """
    Load the clinician detailed ratings file, standardise columns,
    map all labels to English, and collapse subtypes.
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Ground_Truth": "Ground_Truth_Label",
        "Final_Answer": "Predicted_Label",
    })

    for col in ["Ground_Truth_Label", "Predicted_Label", "Category", "Vignette_ID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["Ground_Truth_Label"] = df["Ground_Truth_Label"].apply(map_to_english)
    df["Predicted_Label"] = df["Predicted_Label"].apply(map_to_english)

    # Category-aware harmonisation for "A different diagnosis":
    # - Mood   → "A different diagnosis not listed above"
    # - Stress → "A diagnosis from a different diagnostic area (...)"
    # - Anxiety stays as "A different diagnosis"
    DIFF_DX_BY_CATEGORY = {
        "Mood": "A different diagnosis not listed above",
        "Stress": "A diagnosis from a different diagnostic area "
                  "(e.g. mood disorders, psychotic disorders, personality disorders)",
    }
    for col in ["Ground_Truth_Label", "Predicted_Label"]:
        mask = df[col] == "A different diagnosis"
        for cat, replacement in DIFF_DX_BY_CATEGORY.items():
            cat_mask = mask & (df["Category"] == cat)
            df.loc[cat_mask, col] = replacement

    df = df.set_index("Vignette_ID")
    return df


# ---------------------------------------------------------------------------
# Vignette-equal-weighted accuracy
# ---------------------------------------------------------------------------

def compute_vignette_weighted_accuracy(df: pd.DataFrame) -> dict:
    """
    Compute accuracy where each vignette contributes equally, regardless
    of how many clinicians rated it.

    For each vignette, compute the proportion of raters who were correct,
    then average across vignettes.  The 95% CI is a t-based confidence
    interval on the mean of per-vignette accuracies.

    Returns dict with keys matching compute_metrics_for_group format for
    easy merging into the metrics sheets.
    """
    from scipy import stats

    df = df.copy()
    df["correct"] = (df["Ground_Truth_Label"] == df["Predicted_Label"]).astype(int)

    per_vignette = df.groupby(df.index).agg(
        n_raters=("correct", "count"),
        n_correct=("correct", "sum"),
    )
    per_vignette["accuracy"] = per_vignette["n_correct"] / per_vignette["n_raters"]

    mean_acc = per_vignette["accuracy"].mean()
    std_acc = per_vignette["accuracy"].std()
    n = len(per_vignette)
    se = std_acc / np.sqrt(n)

    # 95% CI using t-distribution
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_acc - t_crit * se
    ci_upper = mean_acc + t_crit * se

    return {
        "vw_accuracy": round(mean_acc, 4),
        "vw_std": round(std_acc, 4),
        "vw_CI_lower": round(ci_lower, 4),
        "vw_CI_upper": round(ci_upper, 4),
        "vw_CI": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        "vw_median": round(per_vignette["accuracy"].median(), 4),
        "vw_n_vignettes": n,
        "per_vignette": per_vignette,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading clinician ratings from:\n  {CLINICIAN_FILE}\n")
    df = load_clinician_ratings(CLINICIAN_FILE)

    n_clinicians = df["ID"].nunique()
    n_vignettes = df.index.nunique()
    n_rows = len(df)

    print(f"Clinicians:       {n_clinicians}")
    print(f"Unique vignettes: {n_vignettes}")
    print(f"Total ratings:    {n_rows}")
    print(f"Categories:       {sorted(df['Category'].unique())}")

    # --- Sanity check ---
    print(f"\nUnique Ground_Truth labels ({df['Ground_Truth_Label'].nunique()}):")
    for lbl in sorted(df["Ground_Truth_Label"].unique()):
        print(f"  {lbl}")

    print(f"\nUnique Predicted labels ({df['Predicted_Label'].nunique()}):")
    for lbl in sorted(df["Predicted_Label"].unique()):
        print(f"  {lbl}")

    all_labels = set(df["Ground_Truth_Label"].unique()) | set(df["Predicted_Label"].unique())
    non_ascii = [l for l in all_labels if any(ord(c) > 127 for c in l)]
    if non_ascii:
        print(f"\n⚠️  WARNING: {len(non_ascii)} labels still contain non-ASCII characters:")
        for lbl in sorted(non_ascii):
            print(f"    {lbl}")
        print("  → These need to be added to MULTILINGUAL_TO_ENGLISH.\n")
    else:
        print("\n✓ All labels successfully mapped to English.\n")

    # ------------------------------------------------------------------
    # 0. Save the harmonised clinician ratings file
    #    Each row = one clinician answer to one vignette, with
    #    Ground_Truth_Label and Predicted_Label in English, collapsed,
    #    and aligned to ENGLISH_CM_LABEL_ORDER.
    #    This file is used downstream for LLM vs clinician kappa.
    # ------------------------------------------------------------------
    harmonised_path = OUTPUT_DIR / "clinicians_harmonised.csv"
    df.to_csv(harmonised_path)
    print(f"Saved harmonised clinician ratings: {harmonised_path}")

    # ------------------------------------------------------------------
    # 0b. Derive majority-vote clinician consensus per vignette
    #     For each vignette, the consensus label is the most frequently
    #     chosen diagnosis across all clinicians who rated it.
    #     Ties (no single mode) → vignette is excluded from kappa.
    #
    #     Output: CSV with columns Vignette_ID, Category,
    #     Ground_Truth_Label, Clinician_Consensus, n_raters, n_votes_for_consensus, is_tie
    #     Saved alongside the harmonised file.
    # ------------------------------------------------------------------
    consensus_records = []
    for vignette_id, vgroup in df.groupby(df.index):
        votes = vgroup["Predicted_Label"].value_counts()
        n_raters = len(vgroup)
        top_count = votes.iloc[0]
        is_tie = (votes == top_count).sum() > 1  # multiple labels tied for first

        consensus_records.append({
            "Vignette_ID": vignette_id,
            "Category": vgroup["Category"].iloc[0],
            "Ground_Truth_Label": vgroup["Ground_Truth_Label"].iloc[0],
            "Clinician_Consensus": votes.index[0] if not is_tie else "",
            "n_raters": n_raters,
            "n_votes_for_consensus": int(top_count),
            "is_tie": is_tie,
        })

    consensus_df = pd.DataFrame(consensus_records).set_index("Vignette_ID")

    n_ties = consensus_df["is_tie"].sum()
    n_with_consensus = len(consensus_df) - n_ties
    print(f"\nClinician consensus: {n_with_consensus} vignettes with majority, "
          f"{n_ties} ties excluded")

    consensus_path = OUTPUT_DIR / "clinician_consensus.csv"
    consensus_df.to_csv(consensus_path)
    print(f"Saved clinician consensus: {consensus_path}")

    # ------------------------------------------------------------------
    # 1. Pooled classification metrics (Overall + per Category)
    #    Kappa here = Cohen's kappa of predictions vs ground truth
    #    (treating the ground truth as one "rater" and the clinician
    #    predictions as the other). This is the same metric used for
    #    the LLM evaluation.
    #
    #    Vignette-equal-weighted accuracy is appended to each sheet
    #    so each vignette contributes equally regardless of rater count.
    # ------------------------------------------------------------------
    sheets: dict[str, pd.DataFrame] = {}

    # Overall
    overall_metrics = compute_metrics_for_group(df)
    vw = compute_vignette_weighted_accuracy(df)
    overall_metrics.update({k: v for k, v in vw.items() if k != "per_vignette"})
    sheets["Overall"] = pd.DataFrame([overall_metrics])

    # Per category
    for cat in sorted(df["Category"].unique()):
        cat_df = df[df["Category"] == cat]
        cat_metrics = compute_metrics_for_group(cat_df)
        vw_cat = compute_vignette_weighted_accuracy(cat_df)
        cat_metrics.update({k: v for k, v in vw_cat.items() if k != "per_vignette"})
        sheets[cat] = pd.DataFrame([cat_metrics])

    # Per-class tables
    sheets["PerClass_Overall"] = compute_per_class_table(df)
    for cat in sorted(df["Category"].unique()):
        sheets[f"PerClass_{cat}"] = compute_per_class_table(
            df[df["Category"] == cat]
        )

    save_excel(
        sheets,
        OUTPUT_DIR / "clinician_classification_metrics.xlsx",
        index=False,
    )

    # ------------------------------------------------------------------
    # Print compact accuracy summary (pooled + vignette-weighted)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLINICIAN ACCURACY SUMMARY")
    print("=" * 70)

    print(f"\n{'Subset':<12} {'Pooled Accuracy':<28} {'Vignette-Weighted Accuracy'}")
    print("-" * 70)

    for subset_name, subset_df in [("Overall", df)] + [
        (cat, df[df["Category"] == cat]) for cat in sorted(df["Category"].unique())
    ]:
        # Pooled
        pooled = compute_accuracy_wilson(subset_df)
        # Vignette-weighted
        vw_sub = compute_vignette_weighted_accuracy(subset_df)

        print(
            f"{subset_name:<12} "
            f"{pooled['accuracy']:.4f} [{pooled['CI_lower']:.4f}, {pooled['CI_upper']:.4f}]"
            f"   "
            f"{vw_sub['vw_accuracy']:.4f} ± {vw_sub['vw_std']:.4f} "
            f"[{vw_sub['vw_CI_lower']:.4f}, {vw_sub['vw_CI_upper']:.4f}]"
        )

    print("-" * 70)
    print("  Pooled: each clinician×vignette row weighted equally")
    print("  Vignette-weighted: each vignette weighted equally (mean ± SD [95% CI])")
    print()

    # ------------------------------------------------------------------
    # 1b. Vignette rating frequency (how often each vignette was rated)
    # ------------------------------------------------------------------
    def _vignette_freq(sub_df: pd.DataFrame) -> pd.DataFrame:
        freq = (
            sub_df.reset_index()
            .groupby("Vignette_ID")
            .agg(
                Category=("Category", "first"),
                Ground_Truth=("Ground_Truth_Label", "first"),
                n_ratings=("ID", "count"),
            )
            .sort_index()
        )
        total = freq["n_ratings"].sum()
        freq["pct_of_total"] = (freq["n_ratings"] / total * 100).round(2)
        return freq

    freq_sheets: dict[str, pd.DataFrame] = {}
    freq_sheets["Overall"] = _vignette_freq(df)

    for cat in sorted(df["Category"].unique()):
        freq_sheets[cat] = _vignette_freq(df[df["Category"] == cat])

    save_excel(
        freq_sheets,
        OUTPUT_DIR / "clinician_vignette_frequencies.xlsx",
        index=True,
    )

    # ------------------------------------------------------------------
    # 2. Per-clinician accuracy
    # ------------------------------------------------------------------
    per_rater_records = []
    for clinician_id, group in df.groupby("ID"):
        row = {"Clinician_ID": clinician_id, "n_vignettes": len(group)}
        acc = compute_accuracy_wilson(group)
        row["accuracy"] = acc["accuracy"]
        row["CI_lower"] = acc["CI_lower"]
        row["CI_upper"] = acc["CI_upper"]
        row["formatted"] = acc["formatted"]

        for cat in sorted(df["Category"].unique()):
            cat_group = group[group["Category"] == cat]
            if len(cat_group) > 0:
                cat_acc = compute_accuracy_wilson(cat_group)
                row[f"accuracy_{cat}"] = cat_acc["accuracy"]
                row[f"n_{cat}"] = len(cat_group)
            else:
                row[f"accuracy_{cat}"] = np.nan
                row[f"n_{cat}"] = 0

        per_rater_records.append(row)

    per_rater_df = pd.DataFrame(per_rater_records).sort_values(
        "accuracy", ascending=False
    )

    summary = {
        "Clinician_ID": "MEAN ± SD",
        "n_vignettes": "",
        "accuracy": per_rater_df["accuracy"].mean(),
        "CI_lower": "",
        "CI_upper": "",
        "formatted": (
            f"{per_rater_df['accuracy'].mean():.4f} ± "
            f"{per_rater_df['accuracy'].std():.4f}"
        ),
    }
    for cat in sorted(df["Category"].unique()):
        col = f"accuracy_{cat}"
        summary[col] = per_rater_df[col].mean()
        summary[f"n_{cat}"] = ""

    per_rater_df = pd.concat(
        [per_rater_df, pd.DataFrame([summary])], ignore_index=True
    )

    save_excel(
        {"Per_Rater_Accuracy": per_rater_df},
        OUTPUT_DIR / "clinician_per_rater_accuracy.xlsx",
        index=False,
    )
    print(f"Per-rater accuracy: mean {per_rater_df['accuracy'].iloc[:-1].mean():.4f} "
          f"± {per_rater_df['accuracy'].iloc[:-1].std():.4f} "
          f"(n={n_clinicians} clinicians)")

    # ------------------------------------------------------------------
    # 3. Vignette-equal-weighted accuracy (save detail file)
    #    Summary already printed above; this saves the per-vignette detail.
    # ------------------------------------------------------------------
    vw_records = []

    vw = compute_vignette_weighted_accuracy(df)
    vw_records.append({
        "Subset": "Overall",
        "mean_accuracy": vw["vw_accuracy"],
        "std": vw["vw_std"],
        "CI_lower": vw["vw_CI_lower"],
        "CI_upper": vw["vw_CI_upper"],
        "CI": vw["vw_CI"],
        "median_accuracy": vw["vw_median"],
        "n_vignettes": vw["vw_n_vignettes"],
    })

    for cat in sorted(df["Category"].unique()):
        vw = compute_vignette_weighted_accuracy(df[df["Category"] == cat])
        vw_records.append({
            "Subset": cat,
            "mean_accuracy": vw["vw_accuracy"],
            "std": vw["vw_std"],
            "CI_lower": vw["vw_CI_lower"],
            "CI_upper": vw["vw_CI_upper"],
            "CI": vw["vw_CI"],
            "median_accuracy": vw["vw_median"],
            "n_vignettes": vw["vw_n_vignettes"],
        })

    save_excel(
        {"Vignette_Weighted_Accuracy": pd.DataFrame(vw_records)},
        OUTPUT_DIR / "clinician_vignette_weighted_accuracy.xlsx",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. Confusion matrices (all labels now in English)
    # ------------------------------------------------------------------
    y_true = df["Ground_Truth_Label"].values
    y_pred = df["Predicted_Label"].values
    all_cm_labels = sorted(set(y_true) | set(y_pred))

    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=all_cm_labels,
        title="Confusion Matrix — Clinicians (all vignettes, pooled)",
        save_path=OUTPUT_DIR / "cm_overall_clinicians.png",
    )

    for cat, cat_group in df.groupby("Category"):
        y_true_cat = cat_group["Ground_Truth_Label"].values
        y_pred_cat = cat_group["Predicted_Label"].values
        cat_labels = sorted(set(y_true_cat) | set(y_pred_cat))
        plot_confusion_matrix(
            y_true_cat,
            y_pred_cat,
            labels=cat_labels,
            title=f"Confusion Matrix — Clinicians / {cat}",
            save_path=OUTPUT_DIR / f"cm_{cat}_clinicians.png",
            category=cat,
            language="english",
        )
        plot_confusion_matrix_pct(
            y_true_cat,
            y_pred_cat,
            labels=cat_labels,
            title=f"Confusion Matrix — Clinicians / {cat}",
            save_path=OUTPUT_DIR / f"cm_{cat}_clinicians_percentage.png",
            category=cat,
            language="english",
        )
        plot_confusion_matrix_pct_fixed(
            y_true_cat,
            y_pred_cat,
            labels=cat_labels,
            title=f"{cat}",
            save_path=OUTPUT_DIR / f"cm_{cat}_clinicians_percentage_fixed_sym_darker2.png",
            category=cat,
            language="english",
            cmap=custom_cmap_cli
        )

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()