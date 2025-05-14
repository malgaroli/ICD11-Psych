import pandas as pd
import re
from pathlib import Path
from LLMModel import LLMModel
from PromptBuilder import PromptBuilder
import json

DEPLOYMENT_TYPE = "computer"

CATEGORIES = ["Mood", "Anxiety", "Stress"]
TOP_K = [1, 2, 3]


def extract_ranked_diagnoses(response):
    diagnoses = []
    pattern1 = r"\*\*\s*(?:1\.|First|Most Likely) Diagnosis:\s*([^\*\n]+)\*\*"
    pattern2 = r"\*\*\s*(?:2\.|Second|Second Most Likely).*?:\s*([^\*\n]+)\*\*"
    pattern3 = r"\*\*\s*(?:3\.|Third|Third Most Likely).*?:\s*([^\*\n]+)\*\*"

    alt_pattern1 = r"(?:1\.|First|Most Likely).*?Diagnosis:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"
    alt_pattern2 = r"(?:2\.|Second|Second Most Likely).*?:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"
    alt_pattern3 = r"(?:3\.|Third|Third Most Likely).*?:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"

    for _, (p1, p2) in enumerate([(pattern1, alt_pattern1), (pattern2, alt_pattern2), (pattern3, alt_pattern3)], 1):
        match = re.search(p1, response, re.IGNORECASE | re.DOTALL) or re.search(p2, response, re.IGNORECASE | re.DOTALL)
        if match:
            diagnosis = re.sub(r'[\*\"\']', '', match.group(1)).strip()
            diagnoses.append(diagnosis)
        else:
            diagnoses.append("NO_MATCH_FOUND")
    return diagnoses


def load_config():
    with open("config.json") as f:
        config_dict = json.load(f)[DEPLOYMENT_TYPE]
    return config_dict


def generate_prompts(data, prompt_builder):
    all_prompts = []
    index_mapping = []
    for category in CATEGORIES:
        df_cat = data[data["Category"] == category]
        if df_cat.empty:
            continue
        prompts = prompt_builder.prepare_cot_prompts(category)
        all_prompts.extend(prompts)
        index_mapping.extend([(category, i) for i in df_cat.index])
    return all_prompts, index_mapping


def evaluate_model_outputs(data, model, all_prompts, responses, index_mapping, llm_model, results_folder):
    import pandas as pd

    # Build base DataFrame
    df = pd.DataFrame(index_mapping, columns=["Category", "DataIndex"])
    df["Prompt"] = [all_prompts[i]["prompt"][0]["content"] for i in range(len(index_mapping))]
    df["Response"] = [list(responses[i].values())[0] for i in range(len(index_mapping))]
    df["Model_Diagnoses"] = df["Response"].apply(extract_ranked_diagnoses)

    # Merge with ground truth
    data_subset = data.loc[df["DataIndex"]].reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, data_subset[["Label", "Vignette ID"]].reset_index(drop=True)], axis=1)
    df["Vignette_ID"] = df["Category"] + " " + df["Vignette ID"].astype(str)

    # Compute Top-k correctness flags
    for k in TOP_K:
        df[f"Top_{k}_Accuracy"] = df.apply(lambda row: row["Label"] in row["Model_Diagnoses"][:k], axis=1)

    # Save detailed outputs
    df_out = df[[
        "Vignette_ID", "Category", "Prompt", "Response",
        "Model_Diagnoses", "Label"
    ] + [f"Top_{k}_Accuracy" for k in TOP_K]].rename(columns={"Response": "Model_Output", "Label": "Ground_Truth_Label"})
    
    df_out.to_csv(results_folder.joinpath(f"ICD11_{llm_model}_ddx_results.csv"), index=False)

    # Save overall performance across categories
    stats = calculate_performance_across_categories(df)
    stats.to_csv(results_folder.joinpath(f"ICD11_{llm_model}_ddx_performance.csv", index=False))




def calculate_performance_across_categories(df):
    # Compute per-category accuracies
    grouped = df.groupby("Category")
    acc_summary = grouped[[f"Top_{k}_Accuracy" for k in TOP_K]].mean().reset_index()
    acc_summary.columns = ["Category"] + [f"top_{k}_accuracy" for k in TOP_K]

    for _, row in acc_summary.iterrows():
        print(f"\n{row['Category']} Accuracies:")
        for k in TOP_K:
            print(f"Top-{k}: {row[f'top_{k}_accuracy']:.2f}")

    # Compute global mean and std
    stats = {}
    for k in TOP_K:
        col = f"top_{k}_accuracy"
        vals = acc_summary[col]
        stats[f"mean_top_{k}_accuracy"] = vals.mean()
        stats[f"std_top_{k}_accuracy"] = vals.std()
        print(f"Top-{k} Average Accuracy: {vals.mean():.2f} (±{vals.std():.2f})")

    # Per-category individual scores also included
    for _, row in acc_summary.iterrows():
        for k in TOP_K:
            stats[f"{row['Category']}_top_{k}_accuracy"] = row[f"top_{k}_accuracy"]

    return pd.DataFrame([stats])

    
def main():
    # set paths
    config_dict = load_config()
    base_bath = Path(config_dict['base_path'])
    results_folder = base_bath.joinpath("results","results_ddx")
    prompt_path = base_bath.joinpath("huggingface_implementation")
    
    # set prompt and pipeline parameters
    llm_model = "llama31"
    prompt_id = 'prompt_ddx'
    language = "en"
    batch_size = 16

    # load vignettes with labels
    data = pd.read_csv(base_bath.joinpath("data","Data_final_updated.csv"))
    
    # Generate prompts for each category
    prompt_builder = PromptBuilder(df_vignettes=data, prompts_path=prompt_path, prompt_id=prompt_id, language=language)
    all_prompts, index_mapping = generate_prompts(data, prompt_builder)
    
    # Initialize model & process prompts in batches
    model_path = Path(config_dict[f"{llm_model}_path"])
    model = LLMModel(model_path)
    responses = model.process_all_batches(all_prompts, batch_size=batch_size)

    # Evaluate model output for top 3 results
    evaluate_model_outputs(data, model, all_prompts, responses, index_mapping, llm_model, results_folder)


if __name__ == "__main__":
    main()
