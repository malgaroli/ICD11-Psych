import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class GuidelineRetriever:
    def __init__(self, chunks_path="guideline_chunks.npy", embeddings_path="guideline_embeddings.npy",
                 model_name='sentence-transformers/sentence-t5-large', top_k=20):
        # error handling
        try:
            self.chunks = np.load(chunks_path, allow_pickle=True)
            self.embeddings = np.load(embeddings_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Error loading files: {e}. Please ensure both .npy files are in the specified paths.")

        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

    def embed_query(self, query):
        # Embed the query
        return self.model.encode([query])[0]

    def retrieve(self, query):
        # Embed the query
        query_embedding = self.embed_query(query)

        # Compute cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get the indices of the top_k most similar chunks
        top_indices = similarities.argsort()[-self.top_k:][::-1]

        # Retrieve and return the most relevant chunks and their similarity scores
        relevant_chunks = [(self.chunks[i], similarities[i]) for i in top_indices]
        return relevant_chunks


if __name__ == "__main__":
    # Sample query (e.g., a clinical vignette)
    sample_query = "Patient shows symptoms of depressive episodes including low mood and lack of interest."

    # Initialize retriever and perform retrieval
    retriever = GuidelineRetriever()
    results = retriever.retrieve(sample_query)

    # Display the most relevant chunks
    print("Most relevant guideline chunks:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\nChunk {i} (Score: {score:.4f}):\n{chunk}")
