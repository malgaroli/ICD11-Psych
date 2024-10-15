# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
# from langchain.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import Chroma
import argparse
import shutil
import os
import numpy as np


Chroma_path = "/Users/xzh/Desktop/CDS_Capstone/"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_document()
    chunks = split_documents(documents)
    add_to_faiss(chunks)


#load pdf file
def load_document():
    document_loader = PyPDFLoader("/Users/xzh/Desktop/CDS_Capstone/data/Mood_Anxiety_Stress.pdf")
    print("document_loader")
    return document_loader.load()


def split_documents(documents: Document):
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=800,
    #     chunk_overlap=80,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    # print("text_splitter")
    # return text_splitter.split_documents(documents)

    # each chunk contains one page of the document
    for document in documents:
       document.metadata["layout_detector"] = "pages"
    return documents
   


def add_to_faiss(chunks: Document):
    embedding_function = get_embedding_function()
    metadatas = [c.metadata for c in chunks]

    
    db = FAISS.from_texts(texts=[c.page_content for c in chunks], 
                         embedding=embedding_function,  # Pass the embedding function directly
                         metadatas=metadatas) 
    db.save_local("Chroma_path")

def calculate_chunk_ids(chunks):
  
    # This will create IDs like in the format of:
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id


def clear_database():
    if os.path.exists(Chroma_path):
        shutil.rmtree(Chroma_path)


if __name__ == "__main__":
    main()