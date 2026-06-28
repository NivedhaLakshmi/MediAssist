import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

all_documents = []

# Read all PDFs in data folder
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        print(f"Loading {file}")

        loader = PyPDFLoader(f"data/{file}")
        documents = loader.load()

        # Store source filename
        for doc in documents:
            doc.metadata["source"] = file

        all_documents.extend(documents)

print(f"Loaded {len(all_documents)} pages")

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_documents)

print(f"Created {len(chunks)} chunks")

# Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)

print("Embeddings stored successfully!")