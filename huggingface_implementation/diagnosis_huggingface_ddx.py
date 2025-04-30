# main_script.py
import pandas as pd
import re
from pathlib import Path
from LLMModel import LLMModel
from PromptBuilder import PromptBuilder
import json

DEPLOYMENT_TYPE = "computer"

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

def main():
    data = pd.read_csv("Data_final_updated.csv")
    prompt_path = Path("cot_text_vicky_ddx")
    config_dict = load_config()
    llm_model = "llama31"
    model_path = Path(config_dict[f"{llm_model}_path"])

    prompt_builder = PromptBuilder(df_vignettes=data, prompts_path=prompt_path)
    all_prompts = []
    index_mapping = []

    # create prompts for different categories
    for category in ["Mood", "Anxiety", "Stress"]:
        df_cat = data[data["Category"] == category]
        if len(df_cat) == 0:
            continue
        prompts = prompt_builder.prepare_cot_prompts(category)
        all_prompts.extend(prompts)
        index_mapping.extend([(category, i) for i in df_cat.index])

    # initialize and run LLM in batches of prompts
    model = LLMModel(model_path)
    responses = model.process_all_batches(all_prompts, batch_size=16)

    all_detailed = []
    all_accuracies = []
    accuracy_results = {
        "Mood": {"top_1_accuracy": [], "top_2_accuracy": [], "top_3_accuracy": []},
        "Anxiety": {"top_1_accuracy": [], "top_2_accuracy": [], "top_3_accuracy": []},
        "Stress": {"top_1_accuracy": [], "top_2_accuracy": [], "top_3_accuracy": []}
    }

    # evaluate different outputs per category
    for category in ["Mood", "Anxiety", "Stress"]:
        indices = [i for i, (cat, _) in enumerate(index_mapping) if cat == category]
        df_cat = data.loc[[index_mapping[i][1] for i in indices]].reset_index(drop=True)
        selected_prompts = [all_prompts[i] for i in indices]
        selected_responses = [list(responses[i].values())[0] for i in indices]

        correct_counts = {1: 0, 2: 0, 3: 0}
        detailed = []

        for i, (_, row) in enumerate(df_cat.iterrows()):
            response = selected_responses[i]
            diagnoses = extract_ranked_diagnoses(response)
            label = row["Label"]
            top_n = {n: label in diagnoses[:n] for n in [1, 2, 3]}
            for n in [1, 2, 3]:
                correct_counts[n] += int(top_n[n])
            detailed.append({
                "Vignette_ID": f"{row['Category']} {row['Vignette ID']}",
                "Category": category,
                "Prompt": selected_prompts[i]['prompt'][0]['content'],
                "Model_Output": response,
                "Model_Diagnoses": diagnoses,
                "Ground_Truth_Label": label,
                "Top_1_Accuracy": top_n[1],
                "Top_2_Accuracy": top_n[2],
                "Top_3_Accuracy": top_n[3]
            })

        total = len(df_cat)
        accs = {f"top_{n}_accuracy": correct_counts[n] / total for n in [1, 2, 3]}
        accs["Category"] = category
        all_accuracies.append(accs)
        all_detailed.extend(detailed)

        for n in [1, 2, 3]:
            accuracy_results[category][f"top_{n}_accuracy"].append(accs[f"top_{n}_accuracy"])

        pd.DataFrame(detailed).to_csv(f"results_{category.lower()}.csv", index=False)

        print(f"\n{category} Accuracies:")
        for n in [1, 2, 3]:
            print(f"Top-{n}: {accs[f'top_{n}_accuracy']:.2f}")

    pd.DataFrame(all_detailed).to_csv("diagnosis_llmmodel_ddx_results.csv", index=False)

    print("\nAverage Accuracies Across Categories:")
    df_acc = pd.DataFrame(all_accuracies)
    stats = {}
    for category in ["Mood", "Anxiety", "Stress"]:
        for n in [1, 2, 3]:
            key = f"{category}_top_{n}_accuracy"
            stats[key] = accuracy_results[category][f"top_{n}_accuracy"][0] if accuracy_results[category][f"top_{n}_accuracy"] else None

    for n in [1, 2, 3]:
        vals = df_acc[f"top_{n}_accuracy"]
        stats[f"mean_top_{n}_accuracy"] = vals.mean()
        stats[f"std_top_{n}_accuracy"] = vals.std()
        print(f"Top-{n} Average Accuracy: {vals.mean():.2f} (±{vals.std():.2f})")

    df_acc.to_csv("mean_accuracies.csv", index=False)
    pd.DataFrame([stats]).to_csv("mean_accuracies_summary.csv", index=False)


if __name__ == "__main__":
    main()
