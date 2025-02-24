import pandas as pd
from langchain_community.llms.ollama import Ollama
from retrieve_relevant_chunks import GuidelineRetriever
import re

retriever = GuidelineRetriever()
model = Ollama(model="llama3.2")
mistral_model = Ollama(model ='mistral')

def load_cot_prompt(category):
    # Map category to corresponding CoT file
    cot_files = {
        "Anxiety": "../cot_text/cot_all.txt",  # cot_anxiety.txt
        "Mood": "../cot_text/cot_all.txt", # cot_mood.txt
        "Stress": "../cot_text/cot_all.txt" # cot_stress.txt
    }
    # Load the CoT content for the given category
    cot_file = cot_files.get(category)
    if cot_file:
        with open(cot_file, 'r') as file:
            return file.read()
    else:
        return "No specific diagnostic steps for this category."


def diagnose_vignette(vignette, retriever, model, category):
    # Retrieve relevant chunks from the guideline
    retrieved_chunks = retriever.retrieve(vignette)
    context_text = "\n\n---\n\n".join([chunk for chunk, _ in retrieved_chunks])

    # Load the category-specific CoT
    cot_text = load_cot_prompt(category)

    prompt = f"""
    You are a professional clinician tasked with diagnosing based on the following vignette. 

    Clinical Guidelines:
    {context_text}

    Vignette:
    {vignette}

    Chain of Thought:
    {cot_text}

    Consider clinical guidelines and chain of thought carefully to reach a thoughtful diagnosis based on the vignette's details.
    If you don't know the answer, just say that you don't know; don't try to make up an answer.
    """

    # print(prompt)
    response = model.invoke(prompt)
    return response


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


def evaluate_model(data, retriever, model, llama_model, mistral_model):
    """Evaluate model performance by comparing predictions with ground truth labels."""
    correct_counts = {"Mood": 0, "Anxiety": 0, "Stress": 0}
    total_counts = {"Mood": 0, "Anxiety": 0, "Stress": 0}
    overall_correct = 0
    total_count = len(data)

    for index, row in data.iterrows():
        category = row["Category"]
        vignette = (f"Referral: {row['Referral']}\n"
                    f"Presenting Symptoms: {row['Presenting Symptoms']}\n"
                    f"Additional Background Information: {row['Additional Background Information']}")
        ground_truth_label = row["Label"]

        model_diagnosis = diagnose_vignette(vignette, retriever, model, category)
        final_diag = extract_diagnosis(model_diagnosis, llama_model)
        similarity_response, score = compare_diagnosis(final_diag, ground_truth_label, mistral_model)

        if score >= 70:
            correct_counts[category] += 1
            overall_correct += 1
        total_counts[category] += 1

        print("------------------------------------------------------------------------------")
        print(f"Vignette ID: {row['Category']} {row['Vignette ID']}")
        print(f"Category: {category}")
        print(f"Vignette Text:\n{vignette}\n")
        print(f"Model Diagnosis: {model_diagnosis}")
        print(f"The extracted Model Diagnosis: {final_diag}")
        print(f"Ground Truth Label: {ground_truth_label}")
        print("The similarity score provided by the model:", score)
        print("The similarity score model reponse:", similarity_response)

    for category in correct_counts:
        category_accuracy = correct_counts[category] / total_counts[category] if total_counts[category] > 0 else 0
        print(f"Accuracy for {category}: {category_accuracy:.2f}")

    overall_accuracy = overall_correct / total_count if total_count > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

    return {
        "Mood": correct_counts["Mood"] / total_counts["Mood"] if total_counts["Mood"] > 0 else 0,
        "Anxiety": correct_counts["Anxiety"] / total_counts["Anxiety"] if total_counts["Anxiety"] > 0 else 0,
        "Stress": correct_counts["Stress"] / total_counts["Stress"] if total_counts["Stress"] > 0 else 0,
        "Overall": overall_accuracy
    }


def main():
    data = pd.read_csv("../Data_final.csv")

    # Filter data to only include Anxiety / Stress / Mood
    category_data = data[data["Category"] == "Mood"]

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
    num_iterations = 8
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        # Reinitialize retriever and models in each iteration
        retriever = GuidelineRetriever()
        model = Ollama(model="llama3.2")
        mistral_model = Ollama(model="mistral")

        # Evaluate the model
        evaluation_results = evaluate_model(
            data,
            retriever,
            model,
            llama_model=model.invoke,
            mistral_model=mistral_model.invoke
        )

        # Collect results for averaging and update highest accuracies
        for key in accuracy_results.keys():
            if key in evaluation_results:  # Ensure the key exists in evaluation_results
                accuracy_results[key].append(evaluation_results[key])
                # Update highest accuracy
                highest_accuracies[key] = max(highest_accuracies[key], evaluation_results[key])

        print(f"Results for Iteration {iteration + 1}: {evaluation_results}")
        print("=" * 80)

    # Calculate average accuracy
    average_results = {
        key: (sum(values) / len(values) if values else 0) for key, values in accuracy_results.items()
    }

    # Print results
    print("Average Accuracy Across All Iterations:")
    for category, avg_accuracy in average_results.items():
        print(f"{category}: {avg_accuracy:.2f}")

    print("\nHighest Accuracy Across All Iterations:")
    for category, highest_accuracy in highest_accuracies.items():
        print(f"{category}: {highest_accuracy:.2f}")


if __name__ == "__main__":
    main()
