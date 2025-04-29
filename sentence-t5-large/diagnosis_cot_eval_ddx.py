import pandas as pd
from langchain_ollama import OllamaLLM
import re
from tqdm import tqdm   

# Model configuration parameters
MODEL_CONFIG = {
    "temperature": 0,
    "top_k": 1,  # Only consider the most likely token
    "top_p": 1,  # Disable nucleus sampling
    "repeat_penalty": 1.0, # Set a fixed number of tokens to predict
    #"num_predict": 128,
    "seed": 42,  # Set a fixed seed if supported by the model
    "do_sample": False  # Always choose the most likely token
}

model = OllamaLLM(model="llama3.1", **MODEL_CONFIG)
mistral_model = OllamaLLM(model='mistral', **MODEL_CONFIG)

def load_cot_prompt(category):
    # Map category to corresponding CoT file
    cot_files = {
        "Anxiety": "cot_text_vicky_ddx/cot_anxiety.txt",  # cot_anxiety.txt
        "Mood": "cot_text_vicky_ddx/cot_mood.txt", # cot_mood.txt
        "Stress": "cot_text_vicky_ddx/cot_stress.txt" # cot_stress.txt
    }
    # Load the CoT content for the given category
    cot_file = cot_files.get(category)
    if cot_file:
        with open(cot_file, 'r') as file:
            return file.read()
    else:
        return "No specific diagnostic steps for this category."


def diagnose_vignette(vignette, model, category):
    """Diagnose using retrieved guidelines and vignette with ranked differential diagnoses."""

    # Load the appropriate prompt template based on category
    cot_text = load_cot_prompt(category)
    
    prompt = f"""
    [ROLE]
    You are a psychiatrist conducting a diagnostic assessment. Read the patient case and follow the instructions to provide a diagnosis. Note: Not all patients will have a diagnosis.

    [PATIENT CASE]
    {vignette}

    [INSTRUCTIONS]
    {cot_text}

    [REQUIRED FORMAT]
    Provide EXACTLY 3 possible diagnoses in a descending order of likelihood:

    1. Most Likely Diagnosis: [diagnosis]
       Reasoning: [2-3 sentences]

    2. Second Most Likely: [diagnosis]
       Reasoning: [1-2 sentences]

    3. Third Most Likely: [diagnosis]
       Reasoning: [1-2 sentences]
    """

    response = model.invoke(prompt)
    return response, prompt


def extract_ranked_diagnoses(response):
    """Extract the ranked diagnoses from the model's response."""
    diagnoses = []
    
    # Pattern 1: "**1. Most Likely Diagnosis: [diagnosis]**"
    pattern1 = r"\*\*\s*(?:1\.|First|Most Likely) Diagnosis:\s*([^\*\n]+)\*\*"
    pattern2 = r"\*\*\s*(?:2\.|Second|Second Most Likely).*?:\s*([^\*\n]+)\*\*"
    pattern3 = r"\*\*\s*(?:3\.|Third|Third Most Likely).*?:\s*([^\*\n]+)\*\*"
    
    # Pattern 2: "1. Most Likely Diagnosis: [diagnosis]"
    alt_pattern1 = r"(?:1\.|First|Most Likely).*?Diagnosis:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"
    alt_pattern2 = r"(?:2\.|Second|Second Most Likely).*?:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"
    alt_pattern3 = r"(?:3\.|Third|Third Most Likely).*?:\s*([^\n]+?)(?=\s*Reasoning:|\n|$)"
    
    # Try both patterns for each position
    for i, (pattern_a, pattern_b) in enumerate([(pattern1, alt_pattern1), 
                                              (pattern2, alt_pattern2), 
                                              (pattern3, alt_pattern3)], 1):
        # Try first pattern style
        match = re.search(pattern_a, response, re.IGNORECASE | re.DOTALL)
        if not match:
            # Try alternative pattern style
            match = re.search(pattern_b, response, re.IGNORECASE | re.DOTALL)
        
        if match:
            diagnosis = match.group(1).strip()
            # Remove any remaining asterisks or quotes
            diagnosis = re.sub(r'[\*"\']', '', diagnosis)
            diagnoses.append(diagnosis)
        else:
            print(f"Warning: Could not find diagnosis {i} in response")
            diagnoses.append("NO_MATCH_FOUND")
    
    return diagnoses


def evaluate_top_n_accuracy(ranked_diagnoses, ground_truth):
    """Calculate top-n accuracy for n=1,2,3."""
    top_n_accuracies = {1: False, 2: False, 3: False}
    
    for n in range(1, 4):
        if ground_truth in ranked_diagnoses[:n]:
            top_n_accuracies[n] = True
    
    return top_n_accuracies


def evaluate_model(data, model, llama_model, mistral_model, category):
    """Evaluate model performance with top-n accuracy."""
    correct_counts = {1: 0, 2: 0, 3: 0}  # For top-1, top-2, top-3 accuracy
    total_count = len(data)
    
    # Create a list to store detailed results
    detailed_results = []

    for index, row in tqdm(data.iterrows(), total=len(data)):
        vignette = (f"Referral: {row['Referral']}\n"
                    f"Presenting Symptoms: {row['Presenting Symptoms']}\n"
                    f"Additional Background Information: {row['Additional Background Information']}")
        ground_truth_label = row["Label"]

        model_diagnosis, prompt = diagnose_vignette(vignette, model, category)
        ranked_diagnoses = extract_ranked_diagnoses(model_diagnosis)
        top_n_accuracies = evaluate_top_n_accuracy(ranked_diagnoses, ground_truth_label)

        # Store detailed results
        detailed_results.append({
            "Vignette_ID": f"{row['Category']} {row['Vignette ID']}",
            "Category": category,
            "Vignette_Text": vignette,
            "Prompt": prompt,
            "Model_Output": model_diagnosis,
            "Model_Diagnoses": ranked_diagnoses,
            "Ground_Truth_Label": ground_truth_label,
            "Top_1_Accuracy": top_n_accuracies[1],
            "Top_2_Accuracy": top_n_accuracies[2],
            "Top_3_Accuracy": top_n_accuracies[3]
        })

        # Update correct counts
        for n in range(1, 4):
            if top_n_accuracies[n]:
                correct_counts[n] += 1

    # Calculate accuracies
    accuracies = {
        f"top_{n}_accuracy": correct_counts[n] / total_count if total_count > 0 else 0
        for n in range(1, 4)
    }

    return accuracies, detailed_results


def save_results_to_csv(results, output_file):
    """Save evaluation results to CSV files with category-specific results."""
    # Save summary results
    summary_rows = []
    for result in results:
        # Base row with iteration
        base_row = {
            "iteration": result["iteration"]
        }
        
        # Add category-specific accuracies
        for category, category_results in result["accuracies"].items():
            for n in range(1, 4):
                base_row[f"{category}_top_{n}_accuracy"] = category_results[f"top_{n}_accuracy"]
        
        # Add overall accuracies across all categories
        for n in range(1, 4):
            total_acc = sum(cat_res[f"top_{n}_accuracy"] for cat_res in result["accuracies"].values())
            base_row[f"overall_top_{n}_accuracy"] = total_acc / len(result["accuracies"]) if result["accuracies"] else 0
            
        summary_rows.append(base_row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_file.replace('.csv', '_summary.csv'), index=False)

    # Save detailed results
    detailed_rows = []
    for result in results:
        for detail in result["detailed_results"]:
            detail["iteration"] = result["iteration"]
            detailed_rows.append(detail)

    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(output_file, index=False)


def main():
    data = pd.read_csv("Data_final_updated.csv")

    # Store accuracies for averaging and highest accuracy
    accuracy_results = {
        "Mood": {"top_1_accuracy": [], "top_2_accuracy": [], "top_3_accuracy": []},
        "Anxiety": {"top_1_accuracy": [], "top_2_accuracy": [], "top_3_accuracy": []},
        "Stress": {"top_1_accuracy": [], "top_2_accuracy": [], "top_3_accuracy": []}
    }
    highest_accuracies = {
        "Mood": {"top_1_accuracy": 0, "top_2_accuracy": 0, "top_3_accuracy": 0},
        "Anxiety": {"top_1_accuracy": 0, "top_2_accuracy": 0, "top_3_accuracy": 0},
        "Stress": {"top_1_accuracy": 0, "top_2_accuracy": 0, "top_3_accuracy": 0}
    }

    # Run evaluation multiple times
    num_iterations = 1
    results = []

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}")
        
        # Initialize combined results for this iteration
        iteration_results = {
            "iteration": iteration + 1,
            "accuracies": {
                "Mood": {"top_1_accuracy": 0, "top_2_accuracy": 0, "top_3_accuracy": 0},
                "Anxiety": {"top_1_accuracy": 0, "top_2_accuracy": 0, "top_3_accuracy": 0},
                "Stress": {"top_1_accuracy": 0, "top_2_accuracy": 0, "top_3_accuracy": 0}
            },
            "detailed_results": []
        }

        # Process each category separately
        for category in ["Mood", "Anxiety", "Stress"]:
            print(f"\nProcessing category: {category}")
            category_data = data[data["Category"] == category]
            if len(category_data) == 0:
                print(f"No data found for category: {category}")
                continue
                
            # Initialize fresh instances for each category
            model = OllamaLLM(model="llama3.1", **MODEL_CONFIG)
            mistral_model = OllamaLLM(model="mistral", **MODEL_CONFIG)

            # Evaluate the model
            accuracies, detailed_results = evaluate_model(
                category_data,
                model,
                llama_model=model.invoke,
                mistral_model=mistral_model.invoke,
                category=category
            )

            # Store category-specific results
            iteration_results["accuracies"][category] = accuracies
            iteration_results["detailed_results"].extend(detailed_results)

            # Update accuracy tracking for this category
            for n in range(1, 4):
                acc_key = f"top_{n}_accuracy"
                accuracy_results[category][acc_key].append(accuracies[acc_key])
                highest_accuracies[category][acc_key] = max(
                    highest_accuracies[category][acc_key],
                    accuracies[acc_key]
                )
            
            print(f"{category} accuracies:")
            for n in range(1, 4):
                print(f"Top-{n}: {accuracies[f'top_{n}_accuracy']:.2f}")

        results.append(iteration_results)

    # Calculate average accuracy per category
    average_results = {}
    for category in ["Mood", "Anxiety", "Stress"]:
        average_results[category] = {
            key: (sum(values) / len(values) if values else 0)
            for key, values in accuracy_results[category].items()
        }

    # Save results to CSV
    save_results_to_csv(results, "diagnosis_cot_eval_ddx_results.csv")

    # Print summary statistics
    print("\nAverage Accuracy Across All Iterations:")
    for category in ["Mood", "Anxiety", "Stress"]:
        print(f"\n{category} Category:")
        for n in range(1, 4):
            print(f"Top-{n} Accuracy: {average_results[category][f'top_{n}_accuracy']:.2f}")

    print("\nHighest Accuracy Across All Iterations:")
    for category in ["Mood", "Anxiety", "Stress"]:
        print(f"\n{category} Category:")
        for n in range(1, 4):
            print(f"Top-{n} Accuracy: {highest_accuracies[category][f'top_{n}_accuracy']:.2f}")


if __name__ == "__main__":
    main()
