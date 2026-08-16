from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_pdf_vectordb(pdf_text: str):
    """
    Create a temporary Chroma vector database
    from an uploaded PDF.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(pdf_text)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
    )

    return vectordb

def search_uploaded_pdf(vectordb, query):
    """
    Search the uploaded PDF.
    """

    docs = vectordb.similarity_search(
        query,
        k=3,
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return docs, context