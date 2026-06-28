from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# Load Llama 3
llm = ChatOllama(
    model="llama3",
    temperature=0
)

# User question
query = input("Ask a question: ")

# Retrieve relevant chunks
docs = vectordb.similarity_search(query, k=3)

for doc in docs:
    print(doc.metadata["source"])

context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
You are a healthcare information assistant.

Using ONLY the provided context, answer the user's question in a clear and complete manner.

If the context contains a definition, explain it fully.
If the context contains symptoms, list them.
If the context contains treatments, summarize them.

Do not mention the context itself.

Context:
{context}

Question:
{query}

Answer:
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response.content)