from langchain_community.vectorstores import Chroma

from services.embedding_service import embedding

LOCAL_SCORE_THRESHOLD = 1.0

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)


def search_local_database(query, k=3):
    """Search the general medical knowledge base."""

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
