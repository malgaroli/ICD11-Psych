# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores import FAISS
import argparse
from get_embedding_function import get_embedding_function

Chroma_path = "/Users/xzh/Desktop/CDS_Capstone/data"

PROMPT_TEMPLATE = """
    You are tasked with diagnosing a stress-related disorder based on the following vignette and diagnostic questionnaire. 
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


# Function to load the questionnaire content from Stress_VignetteA.txt
def load_questionnaire():
    with open("Stress_VignetteA.txt", "r") as file:
        return file.read()


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

def query_rag(vignette: str):
    # reset_model_state()
    # Prepare the DB.
    questionnaire_text = load_questionnaire()
    embedding_function = get_embedding_function()
    db = FAISS.load_local("/Users/xzh/Desktop/CDS_Capstone/Chroma_path", embedding_function, allow_dangerous_deserialization=True) 
    # Get relevant documents
    results = db.similarity_search(vignette, k=50)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, vignette = vignette, questionnaire = questionnaire_text)
    print(prompt)

    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)
    print(response_text)

    sources = [doc.metadata.get("id", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()