from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores import FAISS
import os
Chroma_path = "Chroma_path/"

## Embedding
class EmbeddingFunction(Embeddings):
    def __init__(self):
        super().__init__()  # Call superclass constructor
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

    def embed_query(self, query: str):
        return self.model.encode(query)

    def embed_documents(self, documents: list):
        return self.model.encode(documents)

def get_embedding_function():
    return EmbeddingFunction()

## Load and split the documents
def load_document(filename):
    document_loader = PyPDFLoader(filename)
    return document_loader.load()

def split_documents(documents: Document):
    for document in documents:
       document.metadata["layout_detector"] = "pages"
    return documents

def add_to_faiss(chunks: Document):
    embedding_function = get_embedding_function()
    metadatas = [c.metadata for c in chunks]

    db = FAISS.from_texts(texts=[c.page_content for c in chunks],
                          embedding=embedding_function,
                          metadatas=metadatas)
    if not os.path.exists(Chroma_path):
        os.makedirs(Chroma_path)
    db.save_local(Chroma_path)

def build_database():
    file = "Mood_Anxiety_Stress.pdf"
    documents = load_document(file)
    chunks = split_documents(documents)
    add_to_faiss(chunks)

if __name__ == "__main__":
    build_database()