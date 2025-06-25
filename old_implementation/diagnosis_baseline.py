import pandas as pd
from langchain_community.llms.ollama import Ollama

model = Ollama(model="llama3.2")

def diagnose_baseline(vignette, model):
    # Construct a simple diagnostic prompt without RAG or CoT
    prompt = f"""
    You are a professional clinician tasked with diagnosing based on the following vignette. 
    Consider each diagnostic step carefully to reach a thoughtful diagnosis based on the vignette's details.

    If you don't know the answer, just say that you don't know; don't try to make up an answer.
    
    Vignette:
    {vignette}

    Provide your professional diagnosis.
    """

    # print("Prompt Sent to Model:\n", prompt)

    response = model.invoke(prompt)
    return response


def main():
    data = pd.read_csv("Data_final.csv")

    # Filter data to only include Anxiety / Stress / Mood
    # category_data = data[data["Category"] == "Stress"]

    # Iterate over all vignettes in the Anxiety category
    for index, row in data.iterrows():
        vignette = (f"Referral: {row['Referral']}\n"
                    f"Presenting Symptoms: {row['Presenting Symptoms']}\n"
                    f"Additional Background Information: {row['Additional Background Information']}")
        ground_truth_label = row["Label"]

        diagnosis = diagnose_baseline(vignette, model)

        print(f"\nVignette ID: {row['Category']} {row['Vignette ID']}")
        print(f"Ground Truth Label: {ground_truth_label}")
        print(f"Model Diagnosis: {diagnosis}\n")
        print()


if __name__ == "__main__":
    main()
