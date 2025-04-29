import pandas as pd
import re
from pathlib import Path
from huggingface_models.LLMModel import LLMModel
from huggingface_models.PromptBuilder import PromptBuilder
import json

DEPLOYMENT_TYPE = "laptop"
def extract_ranked_diagnoses(response):
    """Extract the ranked diagnoses from the model's response."""
    diagnoses = []

    # Pattern 1: **1. Most Likely Diagnosis: [diagnosis]**
    pattern1 = r"\*\*\s*(?:1\.|First|Most Likely) Diagnosis:\s*([^\*\n]+)\*\*"
    pattern2 = r"\*\*\s*(?:2\.|Second|Second Most Likely).*?:\s*([^\*\n]+)\*\*"
    pattern3 = r"\*\*\s*(?:3\.|Third|Third Most Likely).*?:\s*([^\*\n]+)\*\*"

    # Pattern 2: 1. Most Likely Diagnosis: [diagnosis]
    alt_pattern1 = r"(?:1\.|First|Most Likely).*?Diagnosis:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"
    alt_pattern2 = r"(?:2\.|Second|Second Most Likely).*?:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"
    alt_pattern3 = r"(?:3\.|Third|Third Most Likely).*?:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"

    for _, (p1, p2) in enumerate([(pattern1, alt_pattern1), (pattern2, alt_pattern2), (pattern3, alt_pattern3)], 1):
        match = re.search(p1, response, re.IGNORECASE | re.DOTALL) or re.search(p2, response, re.IGNORECASE | re.DOTALL)
        if match:
            diagnosis = re.sub(r'[\*"\']', '', match.group(1).strip())
            diagnoses.append(diagnosis)
        else:
            diagnoses.append("NO_MATCH_FOUND")
    return diagnoses

def evaluate_model(df, model_path, category, prompt_path, batch_size=16):
    # Prepare prompts   
    prompt_builder = PromptBuilder(df_vignettes=df, prompts_path=prompt_path)
    prompts = prompt_builder.prepare_cot_prompts(category)

    # Initialize model
    model = LLMModel(Path(model_path))

    # Process prompts
    responses = model.process_all_batches(prompts, batch_size=batch_size)

    correct_counts = {1: 0, 2: 0, 3: 0}
    detailed = []

    for i, (_, row) in enumerate(df.iterrows()):
        response = list(responses[i].values())[0]
        diagnoses = extract_ranked_diagnoses(response)
        label = row["Label"]
        top_n = {n: label in diagnoses[:n] for n in [1, 2, 3]}
        for n in [1, 2, 3]:
            correct_counts[n] += int(top_n[n])
        detailed.append({
            "Vignette_ID": f"{row['Category']} {row['Vignette ID']}",
            "Category": category,
            "Prompt": prompts[i]['prompt'][0]['content'],
            "Model_Output": response,
            "Model_Diagnoses": diagnoses,
            "Ground_Truth_Label": label,
            "Top_1_Accuracy": top_n[1],
            "Top_2_Accuracy": top_n[2],
            "Top_3_Accuracy": top_n[3]
        })

    total = len(df)
    accuracies = {f"top_{n}_accuracy": correct_counts[n] / total for n in [1, 2, 3]}
    return accuracies, detailed

def save_results(detailed_results, output_path):
    df = pd.DataFrame(detailed_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def load_config():
    """Loads configurations from JSON files."""
    with open("experiments/multi_agent_aime/config.json") as f:
        config_dict = json.load(f)[DEPLOYMENT_TYPE]

    with open("experiments/multi_agent_aime/tokens.json") as f:
        hf_token = json.load(f)["HF_TOKEN"]

    return config_dict, hf_token

def main():
    data = pd.read_csv("Data_final_updated.csv")
    prompt_path = Path("cot_text_vicky_ddx")
    config_dict, _ = load_config()

    llm_model = "llama31"  

    model_path = Path(config_dict[f"{llm_model}_path"])

    all_detailed = []
    all_accuracies = []

    for category in ["Mood", "Anxiety", "Stress"]:
        df_cat = data[data["Category"] == category]
        if len(df_cat) == 0:
            continue

        accs, detailed = evaluate_model(df_cat, model_path, category, prompt_path)
        all_detailed.extend(detailed)
        accs["Category"] = category
        all_accuracies.append(accs)

        print(f"\n{category} Accuracies:")
        for n in [1, 2, 3]:
            print(f"Top-{n}: {accs[f'top_{n}_accuracy']:.2f}")

    save_results(all_detailed, "diagnosis_llmmodel_ddx_results.csv")

    print("\nAverage Accuracies Across Categories:")
    df_acc = pd.DataFrame(all_accuracies)
    for n in [1, 2, 3]:
        avg = df_acc[f"top_{n}_accuracy"].mean()
        print(f"Top-{n} Average Accuracy: {avg:.2f}")

if __name__ == "__main__":
    main()
