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
        "Anxiety": "cot_text_vicky_newS/cot_anxiety.txt",  # cot_anxiety.txt
        "Mood": "cot_text_vicky_newS/cot_mood.txt", # cot_mood.txt
        "Stress": "cot_text_vicky_newS/cot_stress.txt" # cot_stress.txt
    }
    # Load the CoT content for the given category
    cot_file = cot_files.get(category)
    if cot_file:
        with open(cot_file, 'r') as file:
            return file.read()
    else:
        return "No specific diagnostic steps for this category."


def diagnose_vignette(vignette, model, category):
    # Load the category-specific CoT
    cot_text = load_cot_prompt(category)

    prompt = f"""
    [ROLE]
    You are a psychiatrist conducting a diagnostic assessment. Read the patient case and follow the instructions to provide a diagnosis. Note: Not all patients will have a diagnosis.

    [PATIENT CASE]
    {vignette}

    [INSTRUCTIONS]
    {cot_text}

    [OUTPUT FORMAT]
    Complete these fields exactly as shown:
    DIAGNOSIS: [single diagnosis/no diagnosis]
    RATIONALE: [one sentence]
    """

    # print(prompt)
    response = model.invoke(prompt)
    return response, prompt


def extract_diagnosis(vignette_text, llama_model):
    """Use the model to extract the diagnosis from the vignette text."""
    prompt = f"Extract the final diagnosis from this text:\n\n{vignette_text}"
    response = llama_model(prompt)
    return response.strip()


def compare_diagnosis(model_diagnosis, ground_truth_label, mistral_model):
    """Use the model to determine if the extracted diagnosis matches the ground truth."""
    prompt = (
        f"Given the diagnosis: '{model_diagnosis}'\n\n"
        f"The ground truth diagnosis is: '{ground_truth_label}'.\n\n"
        f"Does the final diagnosis match the ground truth? Give the similarity score (0-100) and provide reasoning."
        f"Example: 'Score: 85. The diagnoses are very similar because...")

    response = mistral_model(prompt).strip()

    score_match = re.search(r"(\d+)", response)
    if score_match:
        score = int(score_match.group(1))
    else:
        score = -1
        # raise ValueError("Unable to parse similarity score from the response.")
    return response, score


def evaluate_model(data, model, llama_model, mistral_model):
    """Evaluate model performance by comparing predictions with ground truth labels."""
    correct_counts = {"Mood": 0, "Anxiety": 0, "Stress": 0}
    total_counts = {"Mood": 0, "Anxiety": 0, "Stress": 0}
    overall_correct = 0
    total_count = len(data)
    
    # Create a list to store detailed results
    detailed_results = []

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating vignettes"):
        category = row["Category"]
        vignette = (f"Referral: {row['Referral']}\n"
                    f"Presenting Symptoms: {row['Presenting Symptoms']}\n"
                    f"Additional Background Information: {row['Additional Background Information']}")
        ground_truth_label = row["Label"]

        model_diagnosis, prompt = diagnose_vignette(vignette, model, category)
        final_diag = extract_diagnosis(model_diagnosis, llama_model)
        similarity_response, score = compare_diagnosis(final_diag, ground_truth_label, mistral_model)

        # Store detailed results
        detailed_results.append({
            "Vignette_ID": f"{row['Category']} {row['Vignette ID']}",
            "Category": category,
            "Vignette_Text": vignette,
            "prompt": prompt,
            "Model_Diagnosis": model_diagnosis,
            "Extracted_Diagnosis": final_diag,
            "Ground_Truth_Label": ground_truth_label,
            "Similarity_Score": score,
            "Similarity_Response": similarity_response
        })

        if score >= 70:
            correct_counts[category] += 1
            overall_correct += 1
        total_counts[category] += 1

    category_accuracies = {
        category: correct_counts[category] / total_counts[category] if total_counts[category] > 0 else 0
        for category in correct_counts
    }
    overall_accuracy = overall_correct / total_count if total_count > 0 else 0

    return category_accuracies, overall_accuracy, detailed_results


def save_results_to_csv(results, output_file):
    """Save evaluation results to a CSV file."""
    # Save summary results
    summary_rows = []
    for result in results:
        row = {
            "iteration": result["iteration"],
            "overall_accuracy": result["overall_accuracy"],
        }
        row.update(result["category_accuracies"])  # Add category accuracies
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_file.replace('.csv', '_summary_1.csv'), index=False)

    # Save detailed results
    detailed_rows = []
    for result in results:
        for detail in result["detailed_results"]:
            detail["iteration"] = result["iteration"]  # Add iteration number to each detail
            detailed_rows.append(detail)

    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(output_file.replace('.csv', '_detailed_1.csv'), index=False)


def main():
    data = pd.read_csv("Data_final.csv")

    # Store accuracies for averaging and highest accuracy
    accuracy_results = {
        "Mood": [],
        "Anxiety": [],
        "Stress": [],
        "Overall": []
    }
    highest_accuracies = {
        "Mood": 0,
        "Anxiety": 0,
        "Stress": 0,
        "Overall": 0
    }

    # Run evaluation multiple times
    num_iterations = 1
    results = []

    for iteration in range(num_iterations):
        # Initialize combined results for this iteration
        iteration_results = {
            "iteration": iteration + 1,
            "category_accuracies": {
                "Mood": 0,
                "Anxiety": 0,
                "Stress": 0
            },
            "overall_accuracy": 0,
            "detailed_results": []
        }
        
        total_correct = 0
        total_samples = 0       

        # Reinitialize retriever and models in each iteration
        for category in tqdm(["Mood", "Anxiety", "Stress"], desc="Evaluating categories"):
            category_data = data[data["Category"] == category]
            if len(category_data) == 0:
                continue
                
            model = OllamaLLM(model="llama3.1", **MODEL_CONFIG)
            mistral_model = OllamaLLM(model="mistral", **MODEL_CONFIG)

            # Evaluate the model
            category_accuracies, category_overall_accuracy, detailed_results = evaluate_model(
                category_data,
                model,
                llama_model=model.invoke,
                mistral_model=mistral_model.invoke
            )

            # Update iteration results
            iteration_results["category_accuracies"][category] = category_accuracies[category]
            iteration_results["detailed_results"].extend(detailed_results)
            
            # Calculate overall accuracy across all categories
            category_correct = int(category_overall_accuracy * len(category_data))
            total_correct += category_correct
            total_samples += len(category_data)

        # Calculate combined overall accuracy for all categories
        iteration_results["overall_accuracy"] = total_correct / total_samples if total_samples > 0 else 0
        results.append(iteration_results)

        # Update accuracy tracking
        for key in accuracy_results.keys():
            if key in iteration_results["category_accuracies"]:
                accuracy_results[key].append(iteration_results["category_accuracies"][key])
                highest_accuracies[key] = max(highest_accuracies[key], iteration_results["category_accuracies"][key])
            elif key == "Overall":
                accuracy_results[key].append(iteration_results["overall_accuracy"])
                highest_accuracies[key] = max(highest_accuracies[key], iteration_results["overall_accuracy"])

    # Calculate average accuracy
    average_results = {
        key: (sum(values) / len(values) if values else 0) for key, values in accuracy_results.items()
    }

    # Save results to CSV
    save_results_to_csv(results, "diagnosis_cot_eval_results_llama3.1_newS.csv")

    # Print summary statistics
    print("\nAverage Accuracy Across All Iterations:")
    for category, avg_accuracy in average_results.items():
        print(f"{category}: {avg_accuracy:.2f}")

    print("\nHighest Accuracy Across All Iterations:")
    for category, highest_accuracy in highest_accuracies.items():
        print(f"{category}: {highest_accuracy:.2f}")


if __name__ == "__main__":
    main()
