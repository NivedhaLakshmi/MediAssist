from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

LOCAL_SCORE_THRESHOLD = 1.0
PDF_SCORE_THRESHOLD = 2.0

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)


def search_local_database(query, pdf_vectordb=None, k=3):
    # ---------------------------------
    # Search Uploaded PDF First
    # ---------------------------------

    if pdf_vectordb:
        
        print ("\nSearching uploaded PDF")

        pdf_results = pdf_vectordb.similarity_search_with_score(
            query,
            k=k
        )

        pdf_docs = []

        for doc, score in pdf_results:

            print(f"PDF Similarity Score: {score}")

            if score < PDF_SCORE_THRESHOLD:
                pdf_docs.append(doc)

        if pdf_docs:

            context = "\n\n".join(
                doc.page_content for doc in pdf_docs
            )
            
            print("using uploaded PDF")

            return pdf_docs, context, "Uploaded PDF"
    
    print("\nSearching Local Knowledge Base")
    results = vectordb.similarity_search_with_score(query, k=k)

    docs = []

    for doc, score in results:
        print(f"Similarity Score: {score}")

        if score < LOCAL_SCORE_THRESHOLD:
            docs.append(doc)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return docs, context, "Local Knowledge Base"