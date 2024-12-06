import pandas as pd
from langchain_community.llms.ollama import Ollama
from retrieve_relevant_chunks import GuidelineRetriever
import re

prompts = [
    {
        "intro": "You are a licensed medical professional tasked with diagnosing based on the following vignette.",
        "instruction": "Summarize the key clinical features from the vignette. Use step-by-step reasoning to identify potential diagnoses. If you are unsure, respond with 'I don't know' and suggest next steps to clarify the case."
    },
    {
        "intro": "You are a senior clinician known for evidence-based diagnostic reasoning. Your task is to analyze the vignette below and identify a likely diagnosis using clinical guidelines.",
        "instruction": "Highlight important clinical details from the vignette. Provide a thoughtful diagnosis and explain your reasoning clearly. If the information is insufficient, state 'I don't know' and recommend further steps to refine the diagnosis."
    },
    {
        "intro": "You are a medical diagnostician analyzing the following vignette to generate a differential diagnosis.",
        "instruction": "Identify and list possible diagnoses ranked by likelihood. Explain the reasoning and evidence supporting each diagnosis. If the vignette lacks sufficient information, state 'I don't know' and suggest what additional data would be needed."
    },
    {
        "intro": "You are part of a clinical team tasked with diagnosing based on the following vignette. Your insights will contribute to a group discussion.",
        "instruction": "Identify key symptoms and findings. Suggest potential diagnoses with clear reasoning and references to clinical guidelines. Clearly state uncertainties and recommend further actions to address them."
    },
    {
        "intro": "You are a healthcare provider tasked with evaluating the following patient case to provide a diagnosis based on clinical guidelines.",
        "instruction": "Carefully analyze the details to provide a safe and thoughtful diagnosis. If the information is unclear or insufficient, respond with 'I don't know' and suggest next steps (e.g., additional tests or observations). Always prioritize the patient’s safety and best interest in your recommendations."
    },
    {
        "intro": "As a trusted expert in clinical diagnostics, you are tasked with analyzing the following vignette for a potential diagnosis.",
        "instruction": "Dissect the clinical details with precision. Leverage your experience and clinical guidelines to suggest a diagnosis. If the information is incomplete, outline what further steps or data would be needed to proceed."
    },
    {
        "intro": "You are a diagnostic specialist known for solving complex medical cases. Your task is to interpret the vignette and determine the best course of action.",
        "instruction": "Examine the case carefully, considering all possible angles. Use evidence-based practices to form a diagnosis. If faced with uncertainty, transparently discuss the limitations and propose a way forward."
    },
    {
        "intro": "Imagine you are guiding a team of medical trainees through the diagnostic process for the following vignette.",
        "instruction": "Break down the clinical features into understandable segments. Highlight potential diagnoses and the reasoning behind them. If any data is missing, explain its significance and recommend how to obtain it."
    },
    {
        "intro": "As a clinician with a reputation for thoroughness, you are asked to evaluate the vignette and offer your diagnostic insights.",
        "instruction": "Analyze the symptoms and clinical details comprehensively. Present a reasoned diagnosis grounded in current guidelines. If additional context is needed, specify the questions or tests required to fill the gaps."
    },
    {
        "intro": "You are an expert tasked with solving a diagnostic puzzle presented in the following vignette.",
        "instruction": "Evaluate the clinical scenario systematically. Propose a thoughtful diagnosis based on established evidence and guidelines. If the case is ambiguous, discuss possible interpretations and the next best steps for resolution."
    },
    {
        "intro": "You are a primary care physician tasked with identifying potential diagnoses based on the provided patient vignette.",
        "instruction": "Focus on identifying the most likely diagnosis. Use concise reasoning based on the vignette and guidelines. If information is missing, clearly state the additional details required for a confident diagnosis."
    },
    {
        "intro": "As a seasoned specialist, your role is to evaluate the following vignette and provide a thorough differential diagnosis.",
        "instruction": "Carefully examine the clinical presentation and prioritize potential diagnoses by likelihood. Highlight any discrepancies or missing data that could aid in refining the diagnosis."
    },
    {
        "intro": "You are a consultant in a multidisciplinary team tasked with analyzing the following vignette for diagnostic insights.",
        "instruction": "Provide a collaborative perspective by integrating clinical guidelines and cross-specialty reasoning. Recommend additional actions if the vignette lacks critical information."
    },
    {
        "intro": "You are an emergency room physician addressing an acute case described in the following vignette.",
        "instruction": "Quickly evaluate the case details to identify urgent diagnoses. Focus on time-sensitive conditions and outline immediate steps or tests for stabilization and further clarification."
    },
    {
        "intro": "You are a diagnostician reviewing this vignette to identify patterns that match rare or atypical presentations.",
        "instruction": "Explore both common and uncommon diagnoses. If a rare condition is suspected, explain the reasoning and suggest confirmatory steps to rule it in or out."
    },
    {
        "intro": "As a senior clinician mentoring junior staff, you are tasked with reviewing the following vignette to model diagnostic excellence.",
        "instruction": "Explain your diagnostic thought process in a step-by-step manner. Highlight teachable moments and best practices for evaluating complex cases."
    },
    {
        "intro": "You are a global health expert tasked with diagnosing based on this vignette, which may involve region-specific conditions.",
        "instruction": "Consider geographical and epidemiological factors in your diagnosis. If unfamiliar diseases are suspected, suggest resources or experts for further evaluation."
    },
    {
        "intro": "You are a mental health specialist tasked with analyzing the following vignette for a potential psychiatric diagnosis.",
        "instruction": "Focus on evaluating mental health symptoms in the context of the patient’s history. If information is insufficient, recommend key areas to explore further."
    },
    {
        "intro": "You are a pediatrician analyzing this vignette to diagnose a case in a young patient.",
        "instruction": "Pay attention to age-specific symptoms and conditions. Use pediatric guidelines to provide a diagnosis and explain your reasoning clearly."
    },
    {
        "intro": "You are a telemedicine provider tasked with diagnosing based on the following vignette shared remotely by a patient.",
        "instruction": "Assess the available information critically and identify potential diagnoses. Suggest next steps, including in-person evaluation if necessary, to address any gaps in the remote assessment."
    }
]


def generate_prompts(base_intro, base_instruction, model, num_variations):
    """Generate diverse prompts using paraphrasing and other creative variations."""
    prompts = [{"intro": base_intro, "instruction": base_instruction}]  # Include the original prompt

    for i in range(num_variations):
        intro_variation = model.invoke(
            f"Generate a variation of this introduction. Only provide the variation itself without any introductory phrases:\n{base_intro}"
        ).strip()
        instruction_variation = model.invoke(
            f"Generate a variation of this instruction. Only provide the variation itself without any introductory phrases:\n{base_instruction}"
        ).strip()
        prompts.append({"intro": intro_variation, "instruction": instruction_variation})

    return prompts


def diagnose_vignette(vignette, retriever, model, intro, instruction):
    """Diagnose using retrieved guidelines and vignette."""
    retrieved_chunks = retriever.retrieve(vignette)
    context_text = "\n\n---\n\n".join([chunk for chunk, _ in retrieved_chunks])

    prompt = f"""
    {intro}

    Clinical Guidelines:
    {context_text}

    Vignette:
    {vignette}

    {instruction}
    """

    response = model.invoke(prompt)
    return response


def extract_diagnosis(vignette_text, llama_model):
    """Extract the diagnosis from the model's response."""
    prompt = f"Extract the final diagnosis from this text:\n\n{vignette_text}"
    response = llama_model(prompt)
    return response.strip()


def compare_diagnosis(model_diagnosis, ground_truth_label, mistral_model):
    """Compare the model's diagnosis with the ground truth label."""
    prompt = (
        f"Given the diagnosis: '{model_diagnosis}'\n\n"
        f"The ground truth diagnosis is: '{ground_truth_label}'.\n\n"
        f"Does the final diagnosis match the ground truth? Give the similarity score (0-100) and provide reasoning."
    )
    response = mistral_model(prompt).strip()

    score_match = re.search(r"(\d+)", response)
    if score_match:
        score = int(score_match.group(1))
    else:
        score = -1

    return response, score


def evaluate_model_with_prompts(data, retriever, model, llama_model, mistral_model, intro, instruction):
    """Evaluate model performance by comparing predictions with ground truth labels."""
    correct_counts = {category: 0 for category in data["Category"].unique()}
    total_counts = {category: 0 for category in data["Category"].unique()}
    overall_correct = 0
    total_count = len(data)

    for index, row in data.iterrows():
        category = row["Category"]
        vignette = (f"Referral: {row['Referral']}\n"
                    f"Presenting Symptoms: {row['Presenting Symptoms']}\n"
                    f"Additional Background Information: {row['Additional Background Information']}")
        ground_truth_label = row["Label"]

        model_diagnosis = diagnose_vignette(vignette, retriever, model, intro, instruction)
        final_diag = extract_diagnosis(model_diagnosis, llama_model)
        similarity_response, score = compare_diagnosis(final_diag, ground_truth_label, mistral_model)

        if score >= 70:
            correct_counts[category] += 1
            overall_correct += 1
        total_counts[category] += 1

    category_accuracies = {
        category: correct_counts[category] / total_counts[category] if total_counts[category] > 0 else 0
        for category in correct_counts
    }
    overall_accuracy = overall_correct / total_count if total_count > 0 else 0

    return category_accuracies, overall_accuracy


def save_results_to_csv(results, output_file):
    """Save evaluation results to a CSV file."""
    rows = []
    for result in results:
        row = {
            "prompt_label": result["prompt_label"],
            "intro": result["intro"],
            "instruction": result["instruction"],
            "overall_accuracy": result["overall_accuracy"],
        }
        row.update(result["category_accuracies"])  # Add category accuracies
        rows.append(row)

        # Print intro and instruction
        print(f"\nPrompt Label: {result['prompt_label']}")
        print(f"Intro: {result['intro']}")
        print(f"Instruction: {result['instruction']}")

        # Print category accuracies and overall accuracy
        print(f"Category Accuracies:")
        for category, accuracy in result["category_accuracies"].items():
            print(f"  {category}: {accuracy:.2f}")
        print(f"Overall Accuracy: {result['overall_accuracy']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


def main():
    data = pd.read_csv("../Data_final.csv")

    # Filter data for all categories (adjust to your needs)
    category_data = data

    # Initialize retriever
    retriever = GuidelineRetriever()

    # Define the original prompt
    original_prompt = {
        "intro": "You are a professional clinician tasked with diagnosing based on the following vignette.",
        "instruction": (
            "Consider clinical guidelines carefully to reach a thoughtful diagnosis "
            "based on the vignette's details. If you don't know the answer, just say that you don't know; "
            "don't try to make up an answer."
        )
    }

    # Include the original prompt as the first test and add additional prompts
    num_prompts_to_test = 20
    prompts_to_test = [original_prompt] + prompts[:num_prompts_to_test]

    results = []

    for i, prompt in enumerate(prompts_to_test):
        # Reinitialize models to ensure no leakage
        model = Ollama(model="llama3.2")
        mistral_model = Ollama(model="mistral")

        # Prompt details
        label = "Original" if i == 0 else f"Prompt {i}"  # Label the original prompt explicitly
        intro = prompt["intro"]
        instruction = prompt["instruction"]

        print(f"\nEvaluating Prompt: {label}")

        # Evaluate the prompt
        category_accuracies, overall_accuracy = evaluate_model_with_prompts(
            category_data, retriever, model, model.invoke, mistral_model.invoke, intro, instruction
        )

        results.append({
            "prompt_label": label,
            "intro": intro,
            "instruction": instruction,
            "category_accuracies": category_accuracies,
            "overall_accuracy": overall_accuracy,
        })

    # Save results to CSV
    save_results_to_csv(results, "evaluate_different_prompts/evaluation_results.csv")


if __name__ == "__main__":
    main()