import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)


class OllamaServiceError(Exception):
    pass


def generate_answer(prompt):
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise OllamaServiceError(
            f"Could not generate an answer with Ollama model '{OLLAMA_MODEL}'. "
            "Make sure Ollama is running and the model is installed."
        ) from e

    return response.content

def refine_query(query):
    """
    Correct spelling mistakes and improve a medical search query.
    Returns ONLY the refined query.
    """

    prompt = f"""
You are assisting a medical literature search.

Your task is to improve the search query.

Rules:
- Correct spelling mistakes.
- Expand obvious medical abbreviations if necessary.
- Do NOT answer the question.
- Do NOT explain anything.
- Return ONLY the improved search query.

Original Query:
{query}
"""

    try:
        response = llm.invoke(prompt)

    except Exception as e:

        raise OllamaServiceError(
            "Could not refine the medical search query."
        ) from e

    return response.content.strip()
