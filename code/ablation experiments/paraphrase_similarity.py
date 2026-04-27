#!/usr/bin/env python3
"""
Paraphrase Similarity Analysis for ICD-11 Vignette Sensitivity Study

Computes Edit Distance (normalized Levenshtein) and Word Similarity
(1 - Jaccard distance on word sets) between original and paraphrased
vignettes across three paraphrase levels (low, medium, high).

Output:
  - Summary table printed to stdout
  - Per-vignette detail CSV written to ./paraphrase_similarity_results/
  - Summary CSV written to ./paraphrase_similarity_results/
"""

import os
import sys
import math
import csv
import argparse
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def levenshtein(s1: str, s2: str) -> int:
    """Standard dynamic-programming Levenshtein distance (character-level)."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    # Roll two rows to save memory
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)
    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, [0] * (len2 + 1)
    return prev[len2]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """
    Normalised edit distance in [0, 1].
    0 = identical, 1 = completely different.
    Normalised by max(len(s1), len(s2)) so it is length-invariant.
    """
    s1, s2 = str(s1).strip(), str(s2).strip()
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein(s1, s2) / max_len


def word_similarity(s1: str, s2: str) -> float:
    """
    Word-level Jaccard similarity: |intersection| / |union|.
    1 = identical word sets, 0 = no overlap.
    """
    words1 = set(str(s1).lower().split())
    words2 = set(str(s2).lower().split())
    union = words1 | words2
    if not union:
        return 1.0
    return len(words1 & words2) / len(union)


def mean_metrics(texts_orig: list, texts_para: list) -> tuple[float, float]:
    """Return (mean_edit_distance, mean_word_similarity) across paired texts."""
    eds, wss = [], []
    for o, p in zip(texts_orig, texts_para):
        if pd.isna(o) or pd.isna(p):
            continue
        eds.append(normalized_edit_distance(o, p))
        wss.append(word_similarity(o, p))
    if not eds:
        return float("nan"), float("nan")
    return float(np.mean(eds)), float(np.mean(wss))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TEXT_COLS = ["Referral", "Presenting Symptoms", "Additional Background Information"]
ID_COL    = "Vignette ID"


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with fallback encoding detection."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def get_text_cols(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return whichever of 'required' cols actually exist in df."""
    present = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [WARNING] Missing columns in {getattr(df, 'name', '?')}: {missing}")
    return present


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute edit-distance and word-similarity for paraphrased vignettes."
    )
    parser.add_argument("--original",   default="Data_final_updated.csv",
                        help="Path to original vignettes CSV")
    parser.add_argument("--low",        default="Data_final_updated_v1_low.csv")
    parser.add_argument("--medium",     default="Data_final_updated_v1_medium.csv")
    parser.add_argument("--high",       default="Data_final_updated_v1_high.csv")
    parser.add_argument("--outdir",     default="paraphrase_similarity_results",
                        help="Directory for output files")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load files ---
    print("Loading data …")
    try:
        orig = load_csv(args.original)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Original file not found: {args.original}")

    level_paths = {"Low": args.low, "Medium": args.medium, "High": args.high}
    level_dfs   = {}
    for level, path in level_paths.items():
        try:
            level_dfs[level] = load_csv(path)
            print(f"  Loaded {level:6s}: {path}  ({len(level_dfs[level])} rows)")
        except FileNotFoundError:
            sys.exit(f"[ERROR] Paraphrase file not found: {path}")

    print(f"  Original     : {args.original}  ({len(orig)} rows)\n")

    # --- Align on Vignette ID if present, else assume same row order ---
    def align(orig_df, para_df):
        if ID_COL in orig_df.columns and ID_COL in para_df.columns:
            merged = orig_df.merge(para_df, on=ID_COL, suffixes=("_orig", "_para"))
            return merged
        # fallback: assume identical ordering
        n = min(len(orig_df), len(para_df))
        merged = orig_df.iloc[:n].copy().reset_index(drop=True)
        for col in TEXT_COLS:
            if col in para_df.columns:
                merged[f"{col}_para"] = para_df[col].iloc[:n].values
        return merged

    # --- Per-vignette detail rows ---
    detail_rows = []

    # --- Summary table ---
    summary_rows = []

    level_descriptions = {
        "Low":    "Synonym swaps only, same sentence structure and quoted speech preserved",
        "Medium": "Half of non-clinical words replaced, transitions adjusted, quoted speech preserved",
        "High":   "Full rewrite, different sentence structures, information reordered, indirect speech",
    }

    for level in ["Low", "Medium", "High"]:
        para_df = level_dfs[level]
        merged  = align(orig, para_df)

        level_eds, level_wss = [], []

        for col in TEXT_COLS:
            orig_col = col if col in merged.columns else f"{col}_orig"
            para_col = f"{col}_para" if f"{col}_para" in merged.columns else col

            if orig_col not in merged.columns or para_col not in merged.columns:
                print(f"  [WARNING] Cannot find column pair for '{col}' at level {level}. Skipping.")
                continue

            for _, row in merged.iterrows():
                vid = row.get(ID_COL, "N/A")
                o   = str(row[orig_col]) if not pd.isna(row[orig_col]) else ""
                p   = str(row[para_col]) if not pd.isna(row[para_col]) else ""
                ed  = normalized_edit_distance(o, p)
                ws  = word_similarity(o, p)
                level_eds.append(ed)
                level_wss.append(ws)
                detail_rows.append({
                    "Level":             level,
                    "Vignette ID":       vid,
                    "Column":            col,
                    "Edit Distance":     round(ed, 4),
                    "Word Similarity":   round(ws, 4),
                })

        mean_ed = float(np.mean(level_eds)) if level_eds else float("nan")
        mean_ws = float(np.mean(level_wss)) if level_wss else float("nan")

        summary_rows.append({
            "Level":             level,
            "Change":            level_descriptions[level],
            "Edit Distance":     round(mean_ed, 2),
            "Word Similarity":   round(mean_ws, 2),
            "N_pairs":           len(level_eds),
        })

    # --- Print summary table ---
    header = f"{'Level':<8} {'Edit Distance':>14} {'Word Similarity':>16}  Change"
    sep    = "-" * 90
    print(sep)
    print("PARAPHRASE SENSITIVITY ANALYSIS — SUMMARY")
    print(sep)
    print(header)
    print(sep)
    for r in summary_rows:
        print(
            f"{r['Level']:<8} "
            f"{r['Edit Distance']:>14.2f} "
            f"{r['Word Similarity']:>16.2f}  "
            f"{r['Change']}"
        )
    print(sep)
    print()

    # Per-column breakdown
    print("PER-COLUMN BREAKDOWN")
    print(sep)
    print(f"{'Level':<8} {'Column':<38} {'Edit Distance':>14} {'Word Similarity':>16}  N")
    print(sep)
    df_detail = pd.DataFrame(detail_rows)
    if not df_detail.empty:
        grouped = df_detail.groupby(["Level", "Column"]).agg(
            ED_mean=("Edit Distance", "mean"),
            WS_mean=("Word Similarity", "mean"),
            N=("Edit Distance", "count"),
        ).reset_index()
        level_order = {"Low": 0, "Medium": 1, "High": 2}
        grouped["_ord"] = grouped["Level"].map(level_order)
        grouped = grouped.sort_values(["_ord", "Column"]).drop(columns="_ord")
        for _, r in grouped.iterrows():
            print(
                f"{r['Level']:<8} {r['Column']:<38} "
                f"{r['ED_mean']:>14.2f} {r['WS_mean']:>16.2f}  {int(r['N'])}"
            )
    print(sep)

    # --- Write output files ---
    summary_path = os.path.join(args.outdir, "summary_table.csv")
    detail_path  = os.path.join(args.outdir, "per_vignette_detail.csv")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    df_detail.to_csv(detail_path, index=False)

    print(f"\nResults written to:")
    print(f"  {summary_path}")
    print(f"  {detail_path}")


if __name__ == "__main__":
    main()
