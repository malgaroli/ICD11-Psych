from langchain_community.llms.ollama import Ollama
import re
import pandas as pd

def load_vignettes(filepath):
    """Load vignettes, categories, and ground truth labels from the CSV file."""
    data = pd.read_csv(filepath)
    vignettes = {}
    for _, row in data.iterrows():
        # Unique identifier and category for context
        vignette_id = row['Vignette ID']
        category = row['Category']
        # Concatenate the text fields for the vignette content
        vignette = f"{row['Referral']}\n{row['Presenting Symptoms']}\n{row['Additional Background Information']}"
        # Store category, vignette content, and ground truth label
        vignettes[vignette_id] = {
            "category": category,
            "vignette_text": vignette.strip(),
            "label": row['Label']
        }
    return vignettes

llama_model = Ollama(model="llama3.2")

def generate_diagnosis(vignette_text):
    """Use the Ollama model to generate a diagnosis for the vignette."""
    prompt = f"You are tasked with diagnosing based on the following vignette:\n\n{vignette_text}"
    response = llama_model(prompt)
    return response

def extract_diagnosis(vignette_text):
    """Use the model to extract the diagnosis from the vignette text."""
    prompt = f"Extract the final diagnosis from this text:\n\n{vignette_text}"
    response = llama_model(prompt)
    return response.strip()

def compare_diagnosis(model_diagnosis, ground_truth_label):
    """Use the model to determine if the extracted diagnosis matches the ground truth."""
    prompt = (
        f"Given the diagnosis: '{model_diagnosis}'\n\n"
        f"The ground truth diagnosis is: '{ground_truth_label}'.\n\n"
        f"Does the final diagnosis match the ground truth? Answer with 'Yes' or 'No' and provide reasoning."
    )
    response = llama_model(prompt).strip()
    return "yes" in response.lower()

def evaluate_model(vignettes):
    """Evaluate model performance by comparing predictions with ground truth labels."""
    correct_counts = {"Mood": 0, "Anxiety": 0, "Stress": 0}
    total_counts = {"Mood": 0, "Anxiety": 0, "Stress": 0}
    overall_correct = 0
    total_count = len(vignettes)

    for vignette_id, vignette_data in vignettes.items():
        category = vignette_data['category']
        vignette_text = vignette_data['vignette_text']
        ground_truth_label = vignette_data['label']

        model_diagnosis = generate_diagnosis(vignette_text).strip()
        final_diag = extract_diagnosis(model_diagnosis)

        match = compare_diagnosis(final_diag, ground_truth_label)
        if match:
            correct_counts[category] += 1
            overall_correct += 1
        total_counts[category] += 1

        # Print context for each vignette
        print(f"Vignette ID: {vignette_id}")
        print(f"Category: {category}")
        print(f"Vignette Text:\n{vignette_text}\n")
        print(f"Model Diagnosis: {model_diagnosis}")
        print(f"The extracted Model Diagnosis: {final_diag}")
        print(f"Ground Truth Label: {ground_truth_label}")
        print(f"Prediction Match: {'Yes' if match else 'No'}\n")

    # Calculate and print accuracy by category
    for category in correct_counts:
        category_accuracy = correct_counts[category] / total_counts[category] if total_counts[category] > 0 else 0
        print(f"Accuracy for {category}: {category_accuracy:.2f}")

    # Calculate and print overall accuracy
    overall_accuracy = overall_correct / total_count if total_count > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

    return {
        "Mood": correct_counts["Mood"] / total_counts["Mood"] if total_counts["Mood"] > 0 else 0,
        "Anxiety": correct_counts["Anxiety"] / total_counts["Anxiety"] if total_counts["Anxiety"] > 0 else 0,
        "Stress": correct_counts["Stress"] / total_counts["Stress"] if total_counts["Stress"] > 0 else 0,
        "Overall": overall_accuracy
    }


if __name__ == "__main__":
    vignettes = load_vignettes('Data_final.csv')
    results = evaluate_model(vignettes)
    print("Results:", results)
