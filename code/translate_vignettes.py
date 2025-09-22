import os
import json
import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type 
from pathlib import Path

TARGET_TEXT_COLS = [
    "Referral",
    "Presenting Symptoms",
    "Additional Background Information",
]
JOIN_KEYS = ["Category", "Vignette ID"]
DEFAULT_MODEL = "gpt-4.1"# -mini"
MISSING_TOKENS = {"", "na", "n/a", "none", "null", "nan"}


def is_missing(x) -> bool:
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x.strip().lower() in MISSING_TOKENS
    return False


def build_prompt(fields, target_language):
    keys = list(fields.keys())
    keys_str = ", ".join(keys)
    content = "\n\n".join([f"{k}: {fields[k]}" for k in keys])

    return (
        "You are a professional translator.\n"
        f"Translate the following fields from English into {target_language}.\n"
        f"Return ONLY a JSON object with exactly these keys: [{keys_str}].\n"
        "Preserve placeholders like {name}, {date}, {{...}}, and any <tags>.\n"
        "Do not add commentary.\n\n"
        "FIELDS START\n"
        f"{content}\n"
        "FIELDS END"
    )


class RateLimitError(Exception):
    pass


def _is_rate_limit_error(e: Exception) -> bool:
    return "rate limit" in str(e).lower()


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1.5, min=1, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
def translate_fields(client, model, fields, target_language):
    prompt = build_prompt(fields, target_language)
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            # response_format={"type": "json_object"},
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            raise RateLimitError(str(e))
        raise

    out_text = getattr(resp, "output_text", None)
    if not out_text:
        raise RuntimeError("No text returned from model")
    out_text = out_text.replace("```", "").replace("json","")
    parsed = json.loads(out_text)
    return {k: parsed.get(k, "") for k in fields.keys()}


def main(
    base_english: str,
    target_file: str,
    language: str,
    api_key: str = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Fill missing or entirely absent translations for target_file using base_english as source.
    Returns the path to the translated CSV.
    """

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please provide api_key or set OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    df_en = pd.read_csv(base_english, encoding='utf-8')
    df_tgt = pd.read_csv(target_file, encoding='utf-8')

    if "gpt_translated" not in df_tgt.columns:
        df_tgt["gpt_translated"] = False

    df_en_keyed = df_en.set_index(JOIN_KEYS)
    df_tgt_keyed = df_tgt.set_index(JOIN_KEYS)

    filled_count = 0
    new_rows = []

    # Iterate over all rows in the English file (source of truth)
    for key, en_row in df_en_keyed.iterrows():
        if key in df_tgt_keyed.index:
            # Row exists -> only translate missing fields
            row_idx = df_tgt_keyed.index.get_loc(key)
            tgt_row = df_tgt.iloc[row_idx]

            fields_to_do = {
                col: str(en_row[col])
                for col in TARGET_TEXT_COLS
                if is_missing(tgt_row.get(col, None)) and not is_missing(en_row.get(col, None))
            }

            if fields_to_do:
                translations = translate_fields(client, model, fields_to_do, language)
                for col, translated in translations.items():
                    df_tgt.at[df_tgt.index[row_idx], col] = translated
                df_tgt.at[df_tgt.index[row_idx], "gpt_translated"] = True
                filled_count += 1

        else:
            # Row missing entirely -> translate all three fields
            fields_to_do = {col: str(en_row[col]) for col in TARGET_TEXT_COLS if not is_missing(en_row.get(col, None))}
            if fields_to_do:
                print("Process ", key[0], key[1])
                translations = translate_fields(client, model, fields_to_do, language)
                new_row = en_row.to_dict()
                for col, translated in translations.items():
                    new_row[col] = translated
                new_row["gpt_translated"] = True
                # Ensure join keys are set as STRING
                new_row["Category"] = str(key[0])
                new_row["Vignette ID"] = str(key[1])
                new_rows.append(new_row)
                filled_count += 1

    # Append new rows if any
    if new_rows:
        df_tgt = pd.concat([df_tgt, pd.DataFrame(new_rows)], ignore_index=True)

    outpath = target_file.replace(".csv", "_translated.csv")
    df_tgt.to_csv(outpath, index=False, encoding='utf-8')
    print(f"Filled/added {filled_count} rows, saved to {outpath}")
    return outpath

    def load_api_key(path="openAI_key.txt") -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()



if __name__ == "__main__":
    import argparse
    config_dict = load_config(file=Path(__file__).parent.joinpath("config_paths.json"))[DEPLOYMENT_TYPE]
    base_bath = Path(config_dict['base_path'])

    base_english = base_path.joinpath('data','Data_final_updated.csv')
    language = "chinese"
    api_key = load_api_key(path=base_path.parent.joinpath('openAI_key.txt'))
    target_file = base_path.joinpath('data',f'multi-languages/{language}/Data_final_{language}.csv')

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_english", default=base_english, help="Path to Data_final_updated.csv")
    parser.add_argument("--target_file", default=target_file, help="Path to target language CSV")
    parser.add_argument("--language", default=language, help="Target language, e.g. Chinese")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default=api_key, help="OpenAI API key (or env var)")
    args = parser.parse_args()

    main(
        base_english=args.base_english,
        target_file=args.target_file,
        language=args.language,
        api_key=args.api_key,
        model=args.model,
    )
