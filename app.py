import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from services.pubmed_service import search_pubmed


# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥"
)

st.title("🏥 Healthcare RAG Assistant")


# ---------------------------------
# Load Components
# ---------------------------------

@st.cache_resource
def load_components():

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )

    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    return vectordb, llm


vectordb, llm = load_components()


# ---------------------------------
# Chat History
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------
# User Input
# ---------------------------------

question = st.chat_input(
    "Ask a healthcare question..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # ---------------------------------
    # Local Retrieval
    # ---------------------------------

    results = vectordb.similarity_search_with_score(
        question,
        k=3
    )

    docs = []

    for doc, score in results:

        print("Similarity Score:", score)

        if score < 1.0:
            docs.append(doc)

    local_context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # ---------------------------------
    # PubMed Retrieval
    # ---------------------------------

    pubmed_result = []
    pubmed_context = ""

    try:

        pubmed_result = search_pubmed(
            f"{question} review"
        )

        for paper in pubmed_result:

            pubmed_context += f"""
Title:
{paper['title']}

Abstract:
{paper['abstract']}

"""

    except Exception as e:

        print("PubMed Error:")
        print(e)

    # ---------------------------------
    # Context Routing
    # ---------------------------------

    if len(local_context.strip()) > 200:

        context = local_context
        source_used = "Local Knowledge Base"

    else:

        context = pubmed_context
        source_used = "PubMed"

    print(f"Using Source: {source_used}")

    # ---------------------------------
    # Prompt
    # ---------------------------------

    prompt = f"""
You are a healthcare patient education assistant.

Explain medical concepts in simple language.

Use ONLY the provided context.

If insufficient information exists,
say so clearly.

Context:
{context}

Question:
{question}

Answer:
"""

    # ---------------------------------
    # Generate Answer
    # ---------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = llm.invoke(prompt)

                answer = response.content

                st.markdown(answer)

                st.caption(
                    f"Source Used: {source_used}"
                )

            except Exception as e:

                answer = (
                    f"Error generating answer: {str(e)}"
                )

                st.error(answer)

        # ---------------------------------
        # Local Sources
        # ---------------------------------

        if docs:

            with st.expander(
                "Knowledge Base Sources"
            ):

                shown = set()

                for doc in docs:

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    if source not in shown:

                        st.write(source)

                        shown.add(source)

        # ---------------------------------
        # PubMed Sources
        # ---------------------------------

        if pubmed_result:

            with st.expander(
                "PubMed References"
            ):

                for paper in pubmed_result:

                    st.write(
                        f"• {paper['title']}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )