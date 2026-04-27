"""
Vignette Paraphraser for Prompt Sensitivity Analysis (OpenAI API)
=================================================================
Generates low, medium, and high paraphrases of clinical vignettes
using GPT-4o to test diagnostic prediction robustness across
surface-level variation.

Usage:
    python paraphrase_vignettes_openai.py \
        --input_file /path/to/Data_final_updated.csv \
        --output_dir paraphrased_outputs/ \
        --levels low medium high \
        --model gpt-4o \
        --api_key sk-...

Output:
    paraphrased_outputs/
        Data_final_updated_v1_low.csv
        Data_final_updated_v1_medium.csv
        Data_final_updated_v1_high.csv
"""

import os
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_TEXT_COLS = [
    "Referral",
    "Presenting Symptoms",
    "Additional Background Information",
]

DEFAULT_MODEL = "gpt-4o"
MISSING_TOKENS = {"", "na", "n/a", "none", "null", "nan"}

# ---------------------------------------------------------------------------
# Paraphrase prompts and temperatures per level
# ---------------------------------------------------------------------------

PARAPHRASE_PROMPTS = {
    "low": (
        "You are a clinical text editor performing a MINIMAL paraphrase. "
        "Your output must be almost identical to the original. Follow these "
        "rules strictly:\n"
        "1. Keep EVERY sentence exactly as it is, in the same order.\n"
        "2. Only replace a FEW individual words (3-5 per paragraph) with "
        "close synonyms (e.g., 'woman' -> 'female', 'mentioned' -> 'noted', "
        "'suggested' -> 'recommended').\n"
        "3. Do NOT restructure, combine, split, or reorder any sentences.\n"
        "4. Do NOT change any clinical terms, names, ages, dates, or scores.\n"
        "5. Do NOT change quoted speech.\n"
        "6. The output MUST be the same number of sentences and approximately "
        "the same length as the original.\n"
        "7. Return ONLY the JSON object, no commentary."
    ),
    "medium": (
        "You are a clinical text editor performing a MODERATE paraphrase. "
        "The result should be clearly based on the original text with the "
        "same structure, but with more rewording than a minimal edit. "
        "Follow these rules:\n"
        "1. Keep the SAME number of sentences and the SAME order of "
        "information as the original.\n"
        "2. Do NOT combine, split, or reorder sentences.\n"
        "3. Keep sentence openings and transition words similar to the "
        "original (e.g., if a sentence starts with 'She provides the "
        "example...', you may change it to 'She gives the example...' "
        "but not restructure into a completely new sentence).\n"
        "4. Within each sentence, replace roughly HALF of the non-clinical "
        "words and phrases with synonyms or alternative phrasing. Keep the "
        "other half identical to the original.\n"
        "5. You may make small clause-level adjustments (e.g., active to "
        "passive) but the sentence should remain clearly recognizable as "
        "a version of the original.\n"
        "6. Preserve ALL clinical facts, symptoms, names, ages, dates, "
        "diagnoses, temporal details, and assessment scores exactly.\n"
        "7. Keep ALL quoted speech exactly as in the original.\n"
        "8. The output must be approximately the same length as the original "
        "(within 5%).\n"
        "9. Return ONLY the JSON object, no commentary."
    ),
    "high": (
        "You are a clinical text editor performing an EXTENSIVE paraphrase. "
        "Completely rewrite the text as if a different clinician is describing "
        "the same patient from memory. Follow these rules:\n"
        "1. Use entirely different sentence structures and vocabulary "
        "throughout.\n"
        "2. Reorder the presentation of information where clinically "
        "appropriate (e.g., start with presenting symptoms instead of "
        "background, or vice versa).\n"
        "3. Convert all direct quotes to indirect speech with different "
        "wording.\n"
        "4. Vary paragraph structure and grouping of information.\n"
        "5. HOWEVER: preserve ALL clinical facts, symptoms, names, ages, "
        "dates, diagnoses, temporal details, and assessment scores exactly. "
        "Do NOT omit, add, or alter any clinical information.\n"
        "6. The output must be approximately the same length as the original "
        "(within 15%).\n"
        "7. Return ONLY the JSON object, no commentary."
    ),
}

LEVEL_TEMPERATURES = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_missing(x) -> bool:
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x.strip().lower() in MISSING_TOKENS
    return False


def build_prompt(fields: dict, level: str) -> str:
    keys = list(fields.keys())
    keys_str = ", ".join(keys)
    content = "\n\n".join([f"{k}: {fields[k]}" for k in keys])

    return (
        f"{PARAPHRASE_PROMPTS[level]}\n\n"
        f"Return a JSON object with exactly these keys: [{keys_str}].\n"
        f"Each value should be the paraphrased version of the corresponding field.\n\n"
        "FIELDS START\n"
        f"{content}\n"
        "FIELDS END"
    )


def load_api_key(path="openAI_key.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1.5, min=1, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
def paraphrase_fields(client, model: str, fields: dict, level: str, seed: int = 42) -> dict:
    prompt = build_prompt(fields, level)
    temperature = LEVEL_TEMPERATURES[level]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise RateLimitError(str(e))
        raise

    out_text = resp.choices[0].message.content
    if not out_text:
        raise RuntimeError("No text returned from model")

    parsed = json.loads(out_text)
    return {k: parsed.get(k, "") for k in fields.keys()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase clinical vignettes at low/medium/high intensity via OpenAI API."
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to input CSV with vignettes.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="paraphrased_outputs",
        help="Directory for output CSVs.",
    )
    parser.add_argument(
        "--levels", nargs="+", choices=["low", "medium", "high"],
        default=["low", "medium", "high"],
        help="Paraphrase levels to generate.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="OpenAI model to use.",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var, or use --api_key_file).",
    )
    parser.add_argument(
        "--api_key_file", type=str, default=None,
        help="Path to file containing OpenAI API key.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducibility.",
    )

    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key
    if not api_key and args.api_key_file:
        api_key = load_api_key(args.api_key_file)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please provide --api_key, --api_key_file, or set OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_stem = Path(args.input_file).stem

    # Load CSV
    df = pd.read_csv(args.input_file)
    df.columns = df.columns.str.strip()
    logger.info("Loaded %d vignettes from %s", len(df), args.input_file)
    logger.info("Columns: %s", list(df.columns))

    # Validate
    missing = [c for c in TARGET_TEXT_COLS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise SystemExit(1)

    # Generate paraphrases for each level
    for level in args.levels:
        logger.info("=" * 60)
        logger.info("Starting paraphrase level: %s (temp=%.1f)", level.upper(), LEVEL_TEMPERATURES[level])
        logger.info("=" * 60)

        df_level = df.copy()
        total = len(df)
        count = 0
        for idx, row in df.iterrows():
            vignette_id = row.get("Vignette ID", f"row_{idx}")
            logger.info("[%s] [%d/%d] %s", level.upper(), idx + 1, total, vignette_id)

            # Collect non-missing text fields
            fields = {
                col: str(row[col])
                for col in TARGET_TEXT_COLS
                if not is_missing(row.get(col, None))
            }

            if not fields:
                continue

            paraphrased = paraphrase_fields(
                client=client,
                model=args.model,
                fields=fields,
                level=level,
                seed=args.seed,
            )

            for col, text in paraphrased.items():
                df_level.at[idx, col] = text


        output_path = output_dir / f"{input_stem}_v1_{level}.csv"
        df_level.to_csv(output_path, index=False)
        logger.info("Saved %s paraphrases to %s", level, output_path)

    logger.info("Done. All paraphrase levels saved to %s/", output_dir)


if __name__ == "__main__":
    main()