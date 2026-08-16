from datetime import datetime

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
from utils.formatter import format_patient_report_context

# ---------------------------------
# Embedding Model
# ---------------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------
# Persistent Patient Database
# ---------------------------------

patient_vectordb = Chroma(
    persist_directory="patient_db",
    embedding_function=embedding
)


# ---------------------------------
# Text Splitter
# ---------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

PATIENT_SCORE_THRESHOLD = 1.0
# ---------------------------------
# Add Patient Report
# ---------------------------------

def add_patient_report(pdf_text, report_name,report_type):
    """
    Store a patient's uploaded report permanently.
    """
    if not pdf_text.strip():
        print("No text found in report.")
        return
    
    report_id = str(uuid.uuid4())

    chunks = text_splitter.split_text(pdf_text)

    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    documents = []

    for index, chunk in enumerate(chunks):

        documents.append(

            Document(

                page_content=chunk,

                metadata={
                    
                    "report_id": report_id,

                    "report_name": report_name,
                    
                    "report_type": report_type,

                    "upload_date": upload_date,

                    "source": "uploaded_pdf",

                    "chunk": index

                }

            )

        )

    patient_vectordb.add_documents(documents)

    print(f"Stored {len(documents)} chunks from {report_name}")
    
    return report_id

def search_patient_reports(query, k=6):
    """
    Search previously uploaded patient reports.
    """

    results = patient_vectordb.similarity_search_with_score(
        query,
        k=k
    )

    docs = []

    for doc, score in results:

        print(
            f"{doc.metadata.get('report_name', 'Unknown')} "
            f"| Score: {score}"
        )

        docs.append(doc)

    context = format_patient_report_context(docs)

    return docs, context, "Patient Reports"