import pandas as pd
from langchain_community.llms.ollama import Ollama
from retrieve_relevant_chunks import GuidelineRetriever

# Initialize model and retriever
retriever = GuidelineRetriever()
model = Ollama(model="llama3.2")


def diagnose_vignette(vignette, retriever, model):
    # Retrieve relevant chunks from the guideline
    retrieved_chunks = retriever.retrieve(vignette)
    context_text = "\n\n---\n\n".join([chunk for chunk, _ in retrieved_chunks])

    prompt = f"""
    You are a professional clinician tasked with diagnosing based on the following vignette. 
    Use the provided guideline information to make a diagnosis. Please answer as a clinician would, 
    providing a professional assessment and diagnosis. 
    
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Guidelines:
    {context_text}

    Vignette:
    {vignette}

    Provide your professional diagnosis.
    """

    # print("Prompt Sent to Model:\n", prompt)

    response = model.invoke(prompt)
    return response


def main():
    data = pd.read_csv("../Data_final.csv")

    # Filter data to only include Anxiety / Stress / Mood
    category_data = data[data["Category"] == "Stress"]

    # Iterate over all vignettes in the Anxiety category
    for index, row in category_data.iterrows():
        vignette = (f"Referral: {row['Referral']}\n"
                    f"Presenting Symptoms: {row['Presenting Symptoms']}\n"
                    f"Additional Background Information: {row['Additional Background Information']}")
        ground_truth_label = row["Label"]

        diagnosis = diagnose_vignette(vignette, retriever, model)

        print(f"\nVignette ID: {row['Category']} {row['Vignette ID']}")
        print(f"Ground Truth Label: {ground_truth_label}")
        print(f"Model Diagnosis: {diagnosis}\n")
        print()


if __name__ == "__main__":
    main()