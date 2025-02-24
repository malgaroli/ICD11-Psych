# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores import FAISS
import argparse
from get_embedding_function import get_embedding_function
import os

Chroma_path = "/Users/xzh/Desktop/CDS_Capstone/data"
PROMPT_TEMPLATE_INITIAL = """
You are tasked with diagnosing a disorder based on the following vignette and guidelines. 

Context (Guidelines):
{context}

Vignette:
{vignette}

Step 1: Diagnose the disorder into one of the following big categories:
- Stress Disorder
- Anxiety Disorder
- Mood Disorder

Please provide the most appropriate category and briefly explain why based on the guidelines.
"""

PROMPT_TEMPLATE = """
    You are tasked with diagnosing based on the following vignette and diagnostic questionnaire. 
    Use the guidelines provided to inform each of your answers. Do not assume answers, but instead base your reasoning on 
    the evidence provided in the guidelines.


    Context:
    {context}

    Vignette:
    {vignette}

    Diagnostic Questionnaire:
    {questionnaire}

    Instructions:
    Go through the questionnaire step by step. Answer each question carefully, and based on your answer, determine the next step 
    in the decision tree. Use the guidelines to justify every answer.

    Final Diagnosis:
    Please provide your final diagnosis based on the questionnaire and your step-by-step reasoning, using the guidelines as evidence.
    """





def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

# def reset_model_state():
#     global model
#     model = Ollama(model="llama3.2")
def load_questionnaire(category: str):
    file_map = {
        "Stress": "Stress_VignetteA.txt",
        "Anxiety": "Anxiety_Vignette.txt",
        "Mood": "Mood_Vignette.txt"
    }

    # Load the corresponding file
    file_path = file_map.get(category)
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    return "Questionnaire not found"

def extract_category_from_response(response_text: str):
    if "Stress Disorder" in response_text:
        return "Stress"
    elif "Anxiety Disorder" in response_text:
        return "Anxiety"
    elif "Mood Disorder" in response_text:
        return "Mood"
    else:
        return "Unknown"
    
def query_rag(vignette: str):
    # reset_model_state()
    # Prepare the DB.

    embedding_function = get_embedding_function()
    db = FAISS.load_local("/Users/xzh/Desktop/CDS_Capstone/Chroma_path", embedding_function, allow_dangerous_deserialization=True) 
    # Get relevant documents
    results = db.similarity_search(vignette, k=50)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    
    prompt_template_initial = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_INITIAL)
    initial_prompt = prompt_template_initial.format(context=context_text, vignette=vignette)
    model = Ollama(model="llama3.2")
    initial_response = model.invoke(initial_prompt)

    category = extract_category_from_response(initial_response) 

    questionnaire_text = load_questionnaire(category)

    if questionnaire_text == "Questionnaire not found":
        print("Error: Could not find the corresponding questionnaire.")
        return

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    full_prompt = prompt_template.format(context=context_text, vignette = vignette, questionnaire = questionnaire_text)

    response_text = model.invoke(full_prompt)
    print(response_text)

    sources = [doc.metadata.get("id", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()