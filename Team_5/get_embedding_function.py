from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.base import Embeddings

class ClinicalBERTEmbeddingFunction(Embeddings):  
    def __init__(self):
        super().__init__()  # Call superclass constructor
        self.model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
    
    def embed_query(self, query: str):
        return self.model.encode(query)
  
    def embed_documents(self, documents: list):
        return self.model.encode(documents)

def get_embedding_function():
    return ClinicalBERTEmbeddingFunction()