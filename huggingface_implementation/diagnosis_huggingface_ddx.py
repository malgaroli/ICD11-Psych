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


def evaluate_model_outputs(data, model, all_prompts, responses, index_mapping, llm_model):
    all_detailed = []
    all_accuracies = []
    accuracy_results = {cat: {f"top_{k}_accuracy": [] for k in TOP_K} for cat in CATEGORIES}

    for category in CATEGORIES:
        indices = [i for i, (cat, _) in enumerate(index_mapping) if cat == category]
        df_cat = data.loc[[index_mapping[i][1] for i in indices]].reset_index(drop=True)
        selected_prompts = [all_prompts[i] for i in indices]
        selected_responses = [list(responses[i].values())[0] for i in indices]

        correct_counts = {k: 0 for k in TOP_K}
        detailed = []

        for i, (_, row) in enumerate(df_cat.iterrows()):
            response = selected_responses[i]
            diagnoses = extract_ranked_diagnoses(response)
            label = row["Label"]
            top_n = {k: label in diagnoses[:k] for k in TOP_K}
            for k in TOP_K:
                correct_counts[k] += int(top_n[k])
            detailed.append({
                "Vignette_ID": f"{row['Category']} {row['Vignette ID']}",
                "Category": category,
                "Prompt": selected_prompts[i]['prompt'][0]['content'],
                "Model_Output": response,
                "Model_Diagnoses": diagnoses,
                "Ground_Truth_Label": label,
                **{f"Top_{k}_Accuracy": top_n[k] for k in TOP_K}
            })

        total = len(df_cat)
        accs = {f"top_{k}_accuracy": correct_counts[k] / total for k in TOP_K}
        accs["Category"] = category
        all_accuracies.append(accs)
        all_detailed.extend(detailed)

        for k in TOP_K:
            accuracy_results[category][f"top_{k}_accuracy"].append(accs[f"top_{k}_accuracy"])

        # pd.DataFrame(detailed).to_csv(f"results_{llm_model}_{category.lower()}.csv", index=False)

        print(f"\n{category} Accuracies:")
        for k in TOP_K:
            print(f"Top-{k}: {accs[f'top_{k}_accuracy']:.2f}")

    return all_detailed, all_accuracies, accuracy_results


def main():
    data = pd.read_csv("Data_final_updated.csv")
    prompt_path = Path("cot_text_vicky_ddx_qualtrics")
    config_dict = load_config()
    llm_model = "llama31"
    model_path = Path(config_dict[f"{llm_model}_path"])

    prompt_builder = PromptBuilder(df_vignettes=data, prompts_path=prompt_path)
    all_prompts, index_mapping = generate_prompts(data, prompt_builder)

    model = LLMModel(model_path)
    responses = model.process_all_batches(all_prompts, batch_size=16)

    all_detailed, all_accuracies, accuracy_results = evaluate_model_outputs(
        data, model, all_prompts, responses, index_mapping, llm_model
    )

    pd.DataFrame(all_detailed).to_csv(f"diagnosis_{llm_model}_ddx_results_qualtrics.csv", index=False)

    print("\nAverage Accuracies Across Categories:")
    df_acc = pd.DataFrame(all_accuracies)
    stats = {}
    for category in CATEGORIES:
        for k in TOP_K:
            key = f"{category}_top_{k}_accuracy"
            stats[key] = accuracy_results[category][f"top_{k}_accuracy"][0] if accuracy_results[category][f"top_{k}_accuracy"] else None

    for k in TOP_K:
        vals = df_acc[f"top_{k}_accuracy"]
        stats[f"mean_top_{k}_accuracy"] = vals.mean()
        stats[f"std_top_{k}_accuracy"] = vals.std()
        print(f"Top-{k} Average Accuracy: {vals.mean():.2f} (±{vals.std():.2f})")

    # df_acc.to_csv("mean_accuracies_qualtrics.csv", index=False)
    pd.DataFrame([stats]).to_csv(f"{llm_model}_mean_accuracies_summary_qualtrics.csv", index=False)


if __name__ == "__main__":
    main()
