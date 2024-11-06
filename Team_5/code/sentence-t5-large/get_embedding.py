from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

class GuidelineEmbedder:
    def __init__(self, pdf_path, model_name='sentence-transformers/sentence-t5-large', chunk_size=512):
        # Initialize model and parameters
        self.document_loader = PyPDFLoader(pdf_path)
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size

    def load_document(self):
        # Load PDF content and join pages into a single string
        document_pages = self.document_loader.load()
        page_texts = [page.page_content for page in document_pages]  # Extract text from each Document
        return " ".join(page_texts)  # Join list of page texts into a single string

    def chunk_text(self, document_text):
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
        return splitter.split_text(document_text)

    def embed_chunks(self, chunks):
        # Embed each chunk and store results
        return self.model.encode(chunks)

    def process_document(self):
        # Load, chunk, and embed document
        document_text = self.load_document()
        chunks = self.chunk_text(document_text)
        embeddings = self.embed_chunks(chunks)
        return chunks, embeddings


if __name__ == "__main__":
    pdf_path = "../../Mood_Anxiety_Stress.pdf"
    output_chunks_path = "guideline_chunks.npy"
    output_embeddings_path = "guideline_embeddings.npy"

    # Initialize embedder and process document
    embedder = GuidelineEmbedder(pdf_path)
    chunks, embeddings = embedder.process_document()

    # Save chunks and embeddings to .npy files
    np.save(output_chunks_path, chunks)
    np.save(output_embeddings_path, embeddings)
    print(f"Chunks saved to {output_chunks_path}")
    print(f"Embeddings saved to {output_embeddings_path}")