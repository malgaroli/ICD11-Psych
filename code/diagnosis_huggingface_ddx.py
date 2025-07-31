import pandas as pd
import re
from pathlib import Path
from LLMModel import LLMModel
from PromptBuilder import PromptBuilder
import json
import torch
import gc

DEPLOYMENT_TYPE = "hpc"

CATEGORIES = ["Mood", "Anxiety", "Stress"]
TOP_K = [1, 2, 3]

def free_memory():
    """Frees GPU memory if CUDA is available."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = reserved_memory - allocated_memory

        print(f"GPU Memory - Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB, "
              f"Reserved: {reserved_memory / 1024**2:.2f} MB, "
              f"Allocated: {allocated_memory / 1024**2:.2f} MB, "
              f"Free: {free_memory / 1024**2:.2f} MB")


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


def load_config(file):
    with open(file) as f:
        config_dict = json.load(f)
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


def evaluate_model_outputs(data, model, all_prompts, responses, index_mapping, llm_model, results_folder, prompt_id):
    # Build results DataFrame
    df = pd.DataFrame(index_mapping, columns=["Category", "DataIndex"])
    df["Prompt"] = [all_prompts[i]["prompt"][0]["content"] for i in range(len(index_mapping))]
    df["Response"] = [list(responses[i].values())[0] for i in range(len(index_mapping))]
    df["Model_Diagnoses"] = df["Response"].apply(extract_ranked_diagnoses)

    # Merge with ground truth
    data_subset = data.loc[df["DataIndex"]].reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, data_subset[["Qualtrics_label", "Vignette ID"]].reset_index(drop=True)], axis=1)
    df["Vignette_ID"] = df["Category"] + " " + df["Vignette ID"].astype(str)

    # Compute Top-k correctness flags
    for k in TOP_K:
        df[f"Top_{k}_Accuracy"] = df.apply(lambda row: row["Qualtrics_label"] in row["Model_Diagnoses"][:k], axis=1)

    # Save detailed outputs
    df_out = df[[
        "Vignette_ID", "Category", "Prompt", "Response",
        "Model_Diagnoses", "Qualtrics_label"
    ] + [f"Top_{k}_Accuracy" for k in TOP_K]].rename(columns={"Response": "Model_Output", "Qualtrics_label": "Ground_Truth_Label"})
    
    output_df_path = results_folder.joinpath(f"ICD11_{llm_model}_{prompt_id}_detailed_results.csv")
    df_out.to_csv(output_df_path, index=False)
    print("Detailed results saved to ", output_df_path)

    # Calculate and save overall performance across categories
    stats = calculate_performance_across_categories(df)
    performance_file_path = results_folder.joinpath(f"ICD11_{llm_model}_{prompt_id}_performance.csv")
    stats.to_csv(performance_file_path, index=False)
    print("Results saved to ", performance_file_path)


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
    # set prompt and pipeline parameters
    pipeline_params = load_config(Path(__file__).parent.joinpath("pipeline_params.json"))
    llm_model      = pipeline_params.get('llm_model', "llama31_8B")
    prompt_id      = pipeline_params.get('prompt_id','prompt_ddx_qualtrics_modified')
    language       = pipeline_params.get('language', 'en')
    language_vignette = pipeline_params.get('language_vignette', 'en')
    batch_size     = pipeline_params.get('batch_size', 1)
    max_new_tokens = pipeline_params.get('max_new_tokens', 512)

    # set paths
    config_dict = load_config(file=Path(__file__).parent.joinpath("config_paths.json"))[DEPLOYMENT_TYPE]
    base_bath = Path(config_dict['base_path'])
    prompt_path = base_bath.joinpath("code")
    results_folder = base_bath.joinpath("results",llm_model,language_vignette)
    results_folder.mkdir(parents=True, exist_ok=True)

    # load vignettes with labels
    
    data = pd.read_csv(base_bath.joinpath("data","multi-languages",f"{language_vignette}",f"Data_final_{language_vignette}.csv"), encoding='utf-8')
    
    # Generate prompts for each category
    prompt_builder = PromptBuilder(df_vignettes=data, prompts_path=prompt_path, prompt_id=prompt_id, language=language)
    all_prompts, index_mapping = generate_prompts(data, prompt_builder)
    
    # Initialize model & process prompts in batches
    free_memory()
    model_path = Path(config_dict[f"{llm_model}_path"])
    model = LLMModel(model_path, max_new_tokens=max_new_tokens)
    responses = model.process_all_batches(all_prompts, batch_size=batch_size)

    # Evaluate model output for top 3 results
    evaluate_model_outputs(data, model, all_prompts, responses, index_mapping, llm_model, results_folder, prompt_id)

if __name__ == "__main__":
    main()
