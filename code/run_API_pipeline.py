import pandas as pd
import re
from pathlib import Path
from LLMModel import LLMModel
from PromptBuilder import PromptBuilder
import json
import torch
import gc
import subprocess

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


def extract_ranked_diagnoses(response, language):
    if not response:
        return ["NO_MATCH_FOUND"]*3

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


def evaluate_model_outputs(data, all_prompts, responses, index_mapping, llm_model, results_folder, prompt_id, language):
    # Build results DataFrame
    df = pd.DataFrame(index_mapping, columns=["Category", "DataIndex"])
    df["Prompt"] = [all_prompts[i]["prompt"][0]["content"] for i in range(len(index_mapping))]
    df["Response"] = [list(responses[i].values())[0] for i in range(len(index_mapping))]
    df["Model_Diagnoses"] = df["Response"].apply(extract_ranked_diagnoses, args=(language,))

    # Set ground truth label (english or translated version)
    gt_label = "Qualtrics_label" if language == "english" else "Translated_label"

    # Merge with ground truth
    data_subset = data.loc[df["DataIndex"]].reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, data_subset[[gt_label, "Vignette ID"]].reset_index(drop=True)], axis=1)
    df["Vignette_ID"] = df["Category"] + " " + df["Vignette ID"].astype(str)

    # Compute Top-k correctness flags
    for k in TOP_K:
        df[f"Top_{k}_Accuracy"] = df.apply(lambda row: row[gt_label] in row["Model_Diagnoses"][:k], axis=1)

    # Save detailed outputs
    df_out = df[[
        "Vignette_ID", "Category", "Prompt", "Response",
        "Model_Diagnoses", gt_label
    ] + [f"Top_{k}_Accuracy" for k in TOP_K]].rename(columns={"Response": "Model_Output", gt_label: "Ground_Truth_Label"})
    
    output_df_path = results_folder.joinpath(f"ICD11_{llm_model}_{prompt_id}_detailed_results.csv")
    df_out.to_csv(output_df_path, index=False, encoding='utf-8')
    print("Detailed results saved to ", output_df_path)

    # Calculate and save overall performance across categories
    stats = calculate_performance_across_categories(df)
    performance_file_path = results_folder.joinpath(f"ICD11_{llm_model}_{prompt_id}_performance.csv")
    stats.to_csv(performance_file_path, index=False, encoding='utf-8')
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

def load_api_key(path="token.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def run_API_LLM(messages, API_KEY, model_version="gpt-5.1/v1.0.0"):
    if "gpt" in model_version:
        return run_gpt(messages, API_KEY, model_version)
    elif ("sonnet" in model_version) or ("opus" in model_version):
        return run_claude(messages, API_KEY, model_version)
    elif "gemini" in model_version:
        return run_gemini(messages, API_KEY, model_version)
    else:
        Warning("No such model")
        
    
def run_gpt(messages, API_KEY, model_version="gpt-5.1/v1.0.0"):
    # gpt-5.1/v1.0.0, gpt-4o/v1.4.0
    from openai import OpenAI
    model_version = "gpt-4o/v1.3.0"
    ## Select your model by (un)commenting the right line.
    endpoint = f"https://kong-api.prod1.nyumc.org/{model_version}"

    deployment = "required-but-not-used-by-openai-lib"
        
    client = OpenAI(
        base_url=endpoint,
        api_key=API_KEY,
        default_headers={"api-key": API_KEY},
    )
    
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages
    )

    assert completion.choices, "Expected at least one choice in the response."
    response = completion.choices[0].message.content
    # response = "test"
    return response

def run_claude(messages, API_KEY, model_version="opus-4.6/v2.0.0"):
    # sonnet-4.5/v2.0.0, opus-4.6/v2.0.0
    from openai import OpenAI

    ## Select your model by (un)commenting the right line.
    endpoint = f"https://kong-api.prod1.nyumc.org/{model_version}"

    deployment = "required-but-not-used-by-openai-lib"
        
    client = OpenAI(
        base_url=endpoint,
        api_key=API_KEY,
        default_headers={"api-key": API_KEY},
    )
    
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages
    )

    assert completion.choices, "Expected at least one choice in the response."
    response = completion.choices[0].message.content
    # response = "test"
    return response

def run_gemini(messages, API_KEY, model_version="gemini-2.5-pro/v1.0.0"):

    # e.g. gemini-2.5-flash-lite/v1.0.0

    from openai import OpenAI

    ## Select your model by (un)commenting the right line.
    endpoint = f"https://kong-api.prod1.nyumc.org/{model_version}"

    deployment = "required-but-not-used-by-openai-lib"
        
    client = OpenAI(
        base_url=endpoint,
        api_key=API_KEY,
        default_headers={"api-key": API_KEY},
    )
    
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages
    )

    assert completion.choices, "Expected at least one choice in the response."
    response = completion.choices[0].message.content
    # response = "test"
    return response
    
    
def main():
    # set prompt and pipeline parameters
    pipeline_params = load_config(Path(__file__).parent.joinpath("pipeline_params.json"))
    llm_model      = pipeline_params.get('llm_model', "gpt-5.1/v1.0.0")
    prompt_id      = pipeline_params.get('prompt_id','prompt_ddx_qualtrics_modified')
    language       = pipeline_params.get('language', 'english')
    languages_vignette = pipeline_params.get('languages_vignette', 'english')

    # set paths
    config_dict = load_config(file=Path(__file__).parent.joinpath("config_paths.json"))[DEPLOYMENT_TYPE]
    base_path = Path(config_dict['base_path'])
    prompt_path = base_path.joinpath("code")
    results_folder_tmp = "results_new" if language == "english" else "results_languages"

    # Load huggingface api key
    API_KEY = load_api_key(path=base_path.joinpath('token.txt'))
    
    # Check model 
    llm_model = [llm_model] if isinstance(llm_model, str) else llm_model
    for llm in llm_model: 
        # Check language 
        languages_vignette = [languages_vignette] if isinstance(languages_vignette, str) else languages_vignette
        for language_vignette in languages_vignette:
            
            results_folder = base_path.joinpath(results_folder_tmp,llm,language_vignette)
            results_folder.mkdir(parents=True, exist_ok=True)
            # load vignettes with labels
            if language_vignette == "english":
                data_path = base_path.joinpath("data",f"Data_final_updated.csv")
            else:
                data_path = base_path.joinpath("data","multi-languages",f"{language_vignette}",f"Data_final_{language_vignette}.csv")
            data = pd.read_csv(data_path, encoding='utf-8')

            # Generate prompts for each category
            prompt_builder = PromptBuilder(df_vignettes=data, prompts_path=prompt_path, prompt_id=prompt_id, language=language)
            all_prompts, index_mapping = generate_prompts(data, prompt_builder)
            model_version = config_dict[f"{llm}_path"]
            list_responses = []
            # Call llm via API & process prompts in batches
            for vp in all_prompts:
                id = vp['id']
                prompt = vp['prompt']
                
                # call GPT model with prompt
                response = run_API_LLM(prompt, API_KEY, model_version)
                output_json = {
                    'record_id': id, 
                    'prompt': prompt[0]['content'],
                    'output': response}
                list_responses.append({id : response})

                # Save a backup for each run
                backup_folder = results_folder.joinpath(f"{llm}_backup")
                backup_folder.mkdir(exist_ok=True)
                df_backup = pd.DataFrame([output_json])
                df_backup.to_csv(backup_folder.joinpath(f"zeroShot_{llm}_{id}.csv"), index=False)

            # Evaluate model output for top 3 results
            evaluate_model_outputs(data, all_prompts, list_responses, index_mapping, llm, results_folder, prompt_id, language)
        

if __name__ == "__main__":
    main()
