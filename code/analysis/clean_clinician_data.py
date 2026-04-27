"""
Clean WHO Clinicians Dataset
─────────────────────────────
• Drops participants with no diagnostic data (both P1_Dx1 and P2_Dx1 missing)
• Reshapes wide → long (one vignette per row)
• Saves cleaned dataset and demographics dataset
"""

import pandas as pd
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Configuration — update paths here
# ---------------------------------------------------------------------------
def load_config(file):
    with open(file) as f:
        config_dict = json.load(f)
    return config_dict

config_dict = load_config(file=Path(__file__).parents[1].joinpath("config_paths.json"))["hpc"]
BASE_PATH = Path(config_dict['base_path'])
RESULTS_FOLDER = BASE_PATH / "results_Apr26" / "clinicians" / "cleaned"
CLINICIAN_FILE = BASE_PATH / "results_Apr26" / "clinicians" / "raw" / "WHO_Clinicians_Dataset.xlsx"


check_missingness_1 = "P1_Final_Dx"
check_missingness_2 = "P2_Final_Dx"

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
CATEGORY_MAP = {1: "Stress", 2: "Mood", 3: "Anxiety"}
LANGUAGE_MAP = {1: "English", 2: "Spanish", 3: "Japanese", 4: "Chinese", 5: "French", 6: "Russian"}
GENDER_MAP = {1: "Male", 2: "Female", 3: "Other"}
PROFESSION_MAP = {
    1: "Medicine",
    2: "Psychology",
    3: "Nursing",
    4: "Social Work",
    5: "Other",
    6: "Counseling",
    7: "Sex Therapy",
    8: "Speech Therapy",
}
REGION_MAP = {
    "AFRO": "Africa",
    "AMRO-North": "Americas-North",
    "AMRO-South": "Americas-South",
    "AMRO- South": "Americas-South",      # handle variant with space
    "EMRO": "Eastern Mediterranean",
    "EURO": "Europe",
    "SEARO": "South-East Asia",
    "WPRO-Asia": "Western Pacific-Asia",
    "WPRO-Oceania": "Western Pacific-Oceania",
    "N/A": "Other",
}

# ---------------------------------------------------------------------------
# Helper: build vignette ID  e.g. "Stress Vignette 2B"
# ---------------------------------------------------------------------------
def make_vignette_id(category_str: str, raw_vignette: str) -> str:
    """Combine category label with vignette number (strip leading 'V')."""
    num = str(raw_vignette).strip()
    if num.upper().startswith("V"):
        num = num[1:]
    return f"{category_str} Vignette {num}"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_excel(CLINICIAN_FILE)
print(f"  Rows loaded: {len(df)}")

# ---------------------------------------------------------------------------
# Map categorical columns (keep originals for safety, work on copies)
# ---------------------------------------------------------------------------
df["Category"] = df["Study"].map(CATEGORY_MAP)
df["Language"] = df["StudyLanguage"].map(LANGUAGE_MAP)
df["Gender"] = df["GENDER"].map(GENDER_MAP)
df["Profession"] = df["Q9X"].map(PROFESSION_MAP)

# Map Region abbreviations → full labels
raw_region = df["Region"].astype(str).str.strip()
df["Region"] = raw_region.map(REGION_MAP)
# Catch blank cells (NaN) that were not matched by "N/A" key
df["Region"] = df["Region"].fillna("Other")

unmapped = raw_region[df["Region"].isna() & raw_region.notna() & (raw_region != "")].unique()
if len(unmapped):
    print(f"  ⚠  Unmapped Region values: {unmapped.tolist()}")

# ---------------------------------------------------------------------------
# Filter: drop participants where BOTH P1_Dx1 and P2_Dx1 are missing
# ---------------------------------------------------------------------------
both_missing = df[check_missingness_1].isna() & df[check_missingness_2].isna()
n_dropped = both_missing.sum()
print(f"\n  Participants with NO diagnostic data (both {check_missingness_1} & {check_missingness_2} missing): {n_dropped}")
df = df[~both_missing].copy()
print(f"  Rows remaining after filter: {len(df)}")

# Flag rows where only one vignette has data
p1_only_missing = df[check_missingness_1].isna() & df[check_missingness_2].notna()
p2_only_missing = df[check_missingness_2].isna() & df[check_missingness_1].notna()
if p1_only_missing.any():
    ids = df.loc[p1_only_missing, "ID"].tolist()
    print(f"  ⚠  {len(ids)} participant(s) missing {check_missingness_1} only (P2 present): {ids}")
if p2_only_missing.any():
    ids = df.loc[p2_only_missing, "ID"].tolist()
    print(f"  ⚠  {len(ids)} participant(s) missing {check_missingness_2} only (P1 present): {ids}")

# ---------------------------------------------------------------------------
# Reshape: wide → long  (one vignette per row)
# ---------------------------------------------------------------------------
rows_v1 = df[df[check_missingness_1].notna()].copy()
rows_v1["Vignette_ID"] = rows_v1.apply(
    lambda r: make_vignette_id(r["Category"], r["First_Vignette"]), axis=1
)
rows_v1["Ground_Truth"] = rows_v1["P1_Correct_Dx_is"]
rows_v1["Final_Answer"] = rows_v1["P1_Final_Dx"]
rows_v1["First_Answer"] = rows_v1["P1_Dx1"]
rows_v1["First_Answer_Correct"] = rows_v1["P1_Dx1"]

rows_v2 = df[df[check_missingness_2].notna()].copy()
rows_v2["Vignette_ID"] = rows_v2.apply(
    lambda r: make_vignette_id(r["Category"], r["Second_Vignette"]), axis=1
)
rows_v2["Ground_Truth"] = rows_v2["P2_Correct_Dx_is"]
rows_v2["Final_Answer"] = rows_v2["P2_Final_Dx"]
rows_v2["First_Answer"] = rows_v2["P2_Dx1"]
rows_v1["First_Answer_Correct"] = rows_v1["P2_Dx1"]

long = pd.concat([rows_v1, rows_v2], ignore_index=True)
print(f"\n  Rows after reshape (long): {len(long)}")

# ---------------------------------------------------------------------------
# Cleaned dataset
# ---------------------------------------------------------------------------
cleaned_cols = ["ID", "Category", "Vignette_ID", "Language", "Ground_Truth", "Final_Answer", "First_Answer"]
df_cleaned = long[cleaned_cols].copy()

# ---------------------------------------------------------------------------
# Demographics dataset
# ---------------------------------------------------------------------------
demo_cols = [
    "ID", "Category", "Language", "Vignette_ID",
    "Gender", "AGE_CUR", "AGE_REG", "Region",
    "YRS_EXP_CUR", "YRS_EXP_REG", "Profession", "Incomelevel",
]
# Only keep columns that actually exist in the data
demo_cols_present = [c for c in demo_cols if c in long.columns]
missing_demo = set(demo_cols) - set(demo_cols_present)
if missing_demo:
    print(f"  ⚠  Demographics columns not found in data (skipped): {missing_demo}")

df_demo = long[demo_cols_present].copy()

# Rename for consistency with your spec
rename_demo = {
    "AGE_CUR": "Age_Current",
    "AGE_REG": "Age_Registration",
    "YRS_EXP_CUR": "Years_Exp_Current",
    "YRS_EXP_REG": "Years_Exp_Registration",
    "Incomelevel": "Income_Level",
}
df_demo.rename(columns={k: v for k, v in rename_demo.items() if k in df_demo.columns}, inplace=True)

# ---------------------------------------------------------------------------
# Check: do duplicated IDs have inconsistent demographic data?
# ---------------------------------------------------------------------------
demo_check_cols = [c for c in df_demo.columns if c not in ("Vignette_ID", "Category", "Language")]
dup_ids = df_demo.loc[df_demo.duplicated(subset="ID", keep=False), "ID"].unique()

if len(dup_ids):
    print(f"\n  Checking {len(dup_ids)} IDs that appear in multiple rows for demographic consistency …")
    inconsistent = []
    for pid in dup_ids:
        subset = df_demo.loc[df_demo["ID"] == pid, demo_check_cols].drop(columns="ID")
        # Check if all rows for this ID are identical on demographic fields
        if not subset.apply(lambda col: col.nunique(dropna=False) == 1).all():
            differing = [c for c in subset.columns if subset[c].nunique(dropna=False) > 1]
            inconsistent.append((pid, differing))
    if inconsistent:
        print(f"  ⚠  {len(inconsistent)} ID(s) have INCONSISTENT demographics across rows:")
        for pid, cols in inconsistent:
            print(f"      ID {pid}: differs on {cols}")
    else:
        print("  ✓  All duplicate IDs have consistent demographic data across rows.")
else:
    print("\n  No duplicate IDs found — skipping consistency check.")

# ---------------------------------------------------------------------------
# Save — cleaned dataset
# ---------------------------------------------------------------------------
out_cleaned = RESULTS_FOLDER / "clinicians_cleaned.xlsx"
out_demo = RESULTS_FOLDER / "clinicians_demographics.xlsx"
out_debug = RESULTS_FOLDER / "clinicians_debug.xlsx"

df_cleaned.to_excel(out_cleaned, index=False)

# ---------------------------------------------------------------------------
# Build debug sheets
# ---------------------------------------------------------------------------
debug_sheets = {}

# 1) Participants with NO diagnostic data (dropped earlier — use original df)
df_raw = pd.read_excel(CLINICIAN_FILE)
df_raw["Category"] = df_raw["Study"].map(CATEGORY_MAP)
df_raw["Language"] = df_raw["StudyLanguage"].map(LANGUAGE_MAP)
_both_missing = df_raw["P1_Dx1"].isna() & df_raw["P2_Dx1"].isna()
df_no_data = df_raw[_both_missing][["ID", "Category", "Language"]].copy()
df_no_data["Reason"] = "Both P1_Dx1 and P2_Dx1 missing"
debug_sheets["No_Data"] = df_no_data
print(f"\n  Debug — No data: {len(df_no_data)} participants")

# 2) Participants missing one vignette only
partial_rows = []
for mask, missing_col, present_col in [
    (p1_only_missing, "P1_Dx1", "P2_Dx1"),
    (p2_only_missing, "P2_Dx1", "P1_Dx1"),
]:
    if mask.any():
        tmp = df.loc[mask, ["ID", "Category", "Language"]].copy()
        tmp["Missing"] = missing_col
        tmp["Present"] = present_col
        partial_rows.append(tmp)
if partial_rows:
    df_partial = pd.concat(partial_rows, ignore_index=True)
else:
    df_partial = pd.DataFrame(columns=["ID", "Category", "Language", "Missing", "Present"])
debug_sheets["Partial_Missing"] = df_partial
print(f"  Debug — Partial missing: {len(df_partial)} rows")

# 3) Inconsistent demographics across duplicate rows
if inconsistent:
    incon_rows = []
    for pid, cols in inconsistent:
        pid_data = df_demo.loc[df_demo["ID"] == pid].copy()
        pid_data["Differing_Columns"] = ", ".join(cols)
        incon_rows.append(pid_data)
    df_incon = pd.concat(incon_rows, ignore_index=True)
else:
    df_incon = pd.DataFrame(columns=list(df_demo.columns) + ["Differing_Columns"])
debug_sheets["Inconsistent_Demo"] = df_incon
print(f"  Debug — Inconsistent demographics: {len(df_incon)} rows ({len(inconsistent) if inconsistent else 0} unique IDs)")

# 4) Participants who filled out the study in multiple languages
#    (same ID appears with >1 distinct Language value)
id_langs = df_demo.groupby("ID")["Language"].apply(lambda x: sorted(x.unique())).reset_index()
id_langs.columns = ["ID", "Languages"]
multi_lang = id_langs[id_langs["Languages"].apply(len) > 1].copy()
multi_lang["Num_Languages"] = multi_lang["Languages"].apply(len)
multi_lang["Languages"] = multi_lang["Languages"].apply(lambda x: ", ".join(x))
debug_sheets["Multi_Language"] = multi_lang
print(f"  Debug — Multi-language participants: {len(multi_lang)} IDs")

# 5) Participants who did the study in multiple categories
#    Sub-table A: overall (ID with >1 category)
#    Sub-table B: differentiated by language (ID+Language with >1 category)
id_cats = df_demo.groupby("ID")["Category"].apply(lambda x: sorted(x.unique())).reset_index()
id_cats.columns = ["ID", "Categories"]
multi_cat = id_cats[id_cats["Categories"].apply(len) > 1].copy()
multi_cat["Num_Categories"] = multi_cat["Categories"].apply(len)
multi_cat["Categories"] = multi_cat["Categories"].apply(lambda x: ", ".join(x))

# Also get their languages for context
id_cat_lang = df_demo.groupby("ID").agg(
    Categories=("Category", lambda x: ", ".join(sorted(x.unique()))),
    Languages=("Language", lambda x: ", ".join(sorted(x.unique()))),
).reset_index()
multi_cat_detail = id_cat_lang[id_cat_lang["ID"].isin(multi_cat["ID"])].copy()
debug_sheets["Multi_Category"] = multi_cat_detail

# Sub-table B: per language — which ID+Language combos span multiple categories
id_lang_cats = df_demo.groupby(["ID", "Language"])["Category"].apply(
    lambda x: sorted(x.unique())
).reset_index()
id_lang_cats.columns = ["ID", "Language", "Categories"]
multi_cat_by_lang = id_lang_cats[id_lang_cats["Categories"].apply(len) > 1].copy()
multi_cat_by_lang["Num_Categories"] = multi_cat_by_lang["Categories"].apply(len)
multi_cat_by_lang["Categories"] = multi_cat_by_lang["Categories"].apply(lambda x: ", ".join(x))
debug_sheets["Multi_Cat_by_Language"] = multi_cat_by_lang

print(f"  Debug — Multi-category participants: {len(multi_cat)} IDs overall, "
      f"{len(multi_cat_by_lang)} ID×Language combos")

# 6) Participants who filled out more than 2 vignettes within a category
#    (each participant should rate exactly 2 vignettes per category)
vignette_counts = df_demo.groupby(["ID", "Category", "Language"])["Vignette_ID"].nunique().reset_index()
vignette_counts.columns = ["ID", "Category", "Language", "Num_Vignettes"]
over_2 = vignette_counts[vignette_counts["Num_Vignettes"] > 2].copy()
debug_sheets["Over_2_Vignettes"] = over_2
print(f"  Debug — Participants with >2 vignettes in a category: {len(over_2)} "
      f"({over_2['ID'].nunique() if len(over_2) else 0} unique IDs)")

# Write debug file
with pd.ExcelWriter(out_debug, engine="openpyxl") as writer:
    for sheet_name, sheet_df in debug_sheets.items():
        sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
print(f"  Debug file saved → {out_debug}  ({len(debug_sheets)} sheets)")

# ---------------------------------------------------------------------------
# Save — demographics (multi-sheet)
# ---------------------------------------------------------------------------
# Columns to keep on deduplicated (per-ID) sheets (drop Vignette_ID)
demo_dedup_cols = [c for c in df_demo.columns if c != "Vignette_ID"]

# ---------------------------------------------------------------------------
# Build descriptive statistics tables (two versions for PI decision)
# ---------------------------------------------------------------------------
import numpy as np


def compute_stats(grp: pd.DataFrame, grp_lang: pd.DataFrame = None) -> dict:
    """Compute Table 1 stats.

    Parameters
    ----------
    grp : DataFrame
        Deduplicated to one row per unit of analysis (ID or ID×Language)
        used for N and all demographic variables.
    grp_lang : DataFrame, optional
        If provided, used ONLY for the Language breakdown (deduplicated on
        ID×Language). When None, Language is computed from `grp`.
    """
    n = len(grp)
    stats = {"N": str(n)}

    # --- Region ---
    region_order = [
        "Africa", "Americas-North", "Americas-South", "Eastern Mediterranean",
        "Europe", "South-East Asia", "Western Pacific-Asia",
        "Western Pacific-Oceania", "Other",
    ]
    if "Region" in grp.columns:
        rc = grp["Region"].value_counts()
        for r in region_order:
            cnt = rc.get(r, 0)
            pct = cnt / n * 100 if n else 0
            stats[f"Region: {r}"] = f"{cnt} ({pct:.1f}%)"

    # --- Sex ---
    sex_order = ["Male", "Female", "Other"]
    if "Gender" in grp.columns:
        sc = grp["Gender"].value_counts()
        for s in sex_order:
            cnt = sc.get(s, 0)
            pct = cnt / n * 100 if n else 0
            stats[f"Sex: {s}"] = f"{cnt} ({pct:.1f}%)"

    # --- Discipline (Profession) ---
    disc_order = [
        "Counseling", "Medicine", "Nursing", "Psychology", "Social Work",
        "Occupational Therapy", "Sex Therapy", "Certified Peer Support Worker", "Other",
    ]
    if "Profession" in grp.columns:
        dc = grp["Profession"].value_counts()
        for d in disc_order:
            cnt = dc.get(d, 0)
            pct = cnt / n * 100 if n else 0
            stats[f"Discipline: {d}"] = f"{cnt} ({pct:.1f}%)"

    # --- Age (current) ---
    if "Age_Current" in grp.columns:
        age = grp["Age_Current"].dropna()
        if len(age):
            stats["Age, years"] = f"{age.mean():.2f} ({age.std():.2f})"
        else:
            stats["Age, years"] = "—"

    # --- Years of experience (current) ---
    if "Years_Exp_Current" in grp.columns:
        yrs = grp["Years_Exp_Current"].dropna()
        if len(yrs):
            stats["Years of experience"] = f"{yrs.mean():.2f} ({yrs.std():.2f})"
        else:
            stats["Years of experience"] = "—"

    # --- Languages ---
    lang_src = grp_lang if grp_lang is not None else grp
    n_lang = len(lang_src)
    lang_order = ["Chinese", "English", "French", "Japanese", "Russian", "Spanish"]
    if "Language" in lang_src.columns:
        lc = lang_src["Language"].value_counts()
        lang_note = f"  (N responses = {n_lang})" if grp_lang is not None else ""
        for l in lang_order:
            cnt = lc.get(l, 0)
            pct = cnt / n_lang * 100 if n_lang else 0
            stats[f"Language: {l}"] = f"{cnt} ({pct:.1f}%)"
        if grp_lang is not None:
            stats["Language: N responses*"] = str(n_lang)

    return stats


# ---- Version A: count by ID×Language (multi-language clinicians counted twice)
table1_a = {}
for cat in ["Anxiety", "Mood", "Stress"]:
    cat_data = df_demo.loc[df_demo["Category"] == cat, demo_dedup_cols].drop_duplicates(
        subset=["ID", "Language"], keep="first"
    )
    table1_a[cat] = compute_stats(cat_data)

df_overall_dedup_a = df_demo[demo_dedup_cols].drop_duplicates(subset=["ID", "Language"], keep="first")
table1_a["Total"] = compute_stats(df_overall_dedup_a)

df_table1_a = pd.DataFrame(table1_a)
df_table1_a.index.name = "Variable"

# ---- Version B: N & demographics by unique ID; Language by ID×Language
table1_b = {}
for cat in ["Anxiety", "Mood", "Stress"]:
    cat_all = df_demo.loc[df_demo["Category"] == cat, demo_dedup_cols]
    cat_by_id = cat_all.drop_duplicates(subset="ID", keep="first")
    cat_by_id_lang = cat_all.drop_duplicates(subset=["ID", "Language"], keep="first")
    table1_b[cat] = compute_stats(cat_by_id, grp_lang=cat_by_id_lang)

all_by_id = df_demo[demo_dedup_cols].drop_duplicates(subset="ID", keep="first")
all_by_id_lang = df_demo[demo_dedup_cols].drop_duplicates(subset=["ID", "Language"], keep="first")
table1_b["Total"] = compute_stats(all_by_id, grp_lang=all_by_id_lang)

df_table1_b = pd.DataFrame(table1_b)
df_table1_b.index.name = "Variable"

# Print both to console
print("\n" + "=" * 80)
print("TABLE 1A — by ID×Language (multi-language clinicians counted in each language)")
print("=" * 80)
print(df_table1_a.to_string())

print("\n" + "=" * 80)
print("TABLE 1B — by unique ID (demographics counted once per clinician)")
print("           Language breakdown uses ID×Language, marked with *")
print("=" * 80)
print(df_table1_b.to_string())
print("=" * 80)

# ---------------------------------------------------------------------------
# Write all sheets
# ---------------------------------------------------------------------------
with pd.ExcelWriter(out_demo, engine="openpyxl") as writer:
    # Sheet 1: All ratings (long format, one row per vignette)
    df_demo.to_excel(writer, sheet_name="All_Ratings", index=False)

    # Sheet 2: Overall — one row per clinician (ID×Language), collapsing vignettes
    #          Note: if a clinician did multiple categories, they appear once here
    #          (first occurrence). The sum of per-category Ns may exceed Total N.
    df_overall = df_demo[demo_dedup_cols].drop_duplicates(
        subset=["ID", "Language"], keep="first"
    )
    df_overall.to_excel(writer, sheet_name="Overall", index=False)
    print(f"\n  Demographics — Overall unique clinicians (ID×Language): {len(df_overall)}")

    # Sheets 3-5: One per category — one row per clinician within that category
    for cat in ["Anxiety", "Mood", "Stress"]:
        df_cat = df_demo.loc[df_demo["Category"] == cat, demo_dedup_cols].drop_duplicates(
            subset=["ID", "Language"], keep="first"
        )
        df_cat.to_excel(writer, sheet_name=cat, index=False)
        print(f"  Demographics — {cat} unique clinicians: {len(df_cat)}")

    # Sheet 6: Table 1A — counted by ID×Language
    df_table1_a.to_excel(writer, sheet_name="Table1_IDxLang")

    # Sheet 7: Table 1B — demographics by unique ID, language by ID×Language
    df_table1_b.to_excel(writer, sheet_name="Table1_UniqueID")

# Sanity check: compare Table 1 Ns with sheet row counts
print("\n  Sanity check — Table 1 N vs sheet rows (should match):")
for cat in ["Anxiety", "Mood", "Stress"]:
    cat_dedup = df_demo.loc[df_demo["Category"] == cat, demo_dedup_cols].drop_duplicates(
        subset=["ID", "Language"], keep="first"
    )
    print(f"    {cat}: Table1_IDxLang N = {table1_a[cat]['N']}, Table1_UniqueID N = {table1_b[cat]['N']}, sheet rows = {len(cat_dedup)}")

print(f"\n✅  Cleaned dataset saved  → {out_cleaned}  ({len(df_cleaned)} rows)")
print(f"✅  Demographics saved    → {out_demo}  (7 sheets: All_Ratings, Overall, Anxiety, Mood, Stress, Table1_IDxLang, Table1_UniqueID)")
print(f"✅  Debug file saved      → {out_debug}  ({len(debug_sheets)} sheets: {', '.join(debug_sheets.keys())})")
print("\nDone.")