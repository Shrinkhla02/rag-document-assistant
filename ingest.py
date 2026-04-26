from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load PDFs
def load_docs():
    print("Loading documents...")
    docs = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"docs/{file}")
            docs.extend(loader.load())
    return docs

# Split text
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

# Create vector DB
def create_vector_db(chunks):
    print("Creating vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    print("Vector DB created successfully")

if __name__ == "__main__":
    print("Starting document ingestion...")
    docs = load_docs()
    chunks = split_docs(docs)
    create_vector_db(chunks)