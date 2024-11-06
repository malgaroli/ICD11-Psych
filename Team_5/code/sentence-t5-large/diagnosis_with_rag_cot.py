import pandas as pd
from langchain_community.llms.ollama import Ollama
from retrieve_relevant_chunks import GuidelineRetriever

retriever = GuidelineRetriever()
model = Ollama(model="llama3.2")


def load_cot_prompt(category):
    # Map category to corresponding CoT file
    cot_files = {
        "Anxiety": "../cot_text/cot_anxiety.txt",
        "Mood": "../cot_text/cot_mood.txt",
        "Stress": "../cot_text/cot_stress.txt"
    }
    # Load the CoT content for the given category
    cot_file = cot_files.get(category)
    if cot_file:
        with open(cot_file, 'r') as file:
            return file.read()
    else:
        return "No specific diagnostic steps for this category."


def diagnose_vignette_with_cot(vignette, retriever, model, category):
    # Retrieve relevant chunks from the guideline
    retrieved_chunks = retriever.retrieve(vignette)
    context_text = "\n\n---\n\n".join([chunk for chunk, _ in retrieved_chunks])

    # Load the category-specific CoT
    cot_text = load_cot_prompt(category)

    prompt = f"""
    You are a professional clinician tasked with diagnosing based on the following vignette. 
    Consider clinical guidelines and each diagnostic guidance step carefully to reach a thoughtful diagnosis based on the vignette's details.

    If you don't know the answer, just say that you don't know; don't try to make up an answer.

    Clinical Guidelines:
    {context_text}
    
    Diagnostic Guidance (Chain of Thought): 
    {cot_text}

    Vignette:
    {vignette}
    """

    # print(prompt)
    response = model.invoke(prompt)
    return response


def main():
    # Load vignette data
    data = pd.read_csv("../Data_final.csv")

    # Iterate over all vignettes, handling multiple categories
    for index, row in data.iterrows():
        vignette = (
            f"Referral: {row['Referral']}\n"
            f"Presenting Symptoms: {row['Presenting Symptoms']}\n"
            f"Additional Background Information: {row['Additional Background Information']}"
        )
        category = row["Category"]
        ground_truth_label = row["Label"]

        diagnosis = diagnose_vignette_with_cot(vignette, retriever, model, category)

        print(f"\nVignette ID: {row['Vignette ID']}")
        print(f"Category: {category}")
        print(f"Ground Truth Label: {ground_truth_label}")
        print(f"Model Diagnosis: {diagnosis}\n")
        print()


if __name__ == "__main__":
    main()
