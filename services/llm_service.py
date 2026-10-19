import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

# The default names the model the application was tuned against.
# A 3B model was used earlier and followed the prompt's rules
# unreliably, so the fallback should not quietly select one.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# Ollama defaults to a small context window (~2048 tokens), which
# truncates the report context + rules + history and can make the
# model return nothing. Give it room.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_ctx=OLLAMA_NUM_CTX,
)


class OllamaServiceError(Exception):
    pass


def _build_messages(prompt, system_prompt=None):
    if system_prompt:
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

    return prompt


def stream_answer(prompt, system_prompt=None):
    """Yield the answer in pieces as the model produces them.

    Streamlit renders each piece as it arrives, so the patient sees
    the answer forming instead of waiting on a blank spinner. The
    total time is unchanged - only the perceived wait.
    """

    messages = _build_messages(prompt, system_prompt)

    produced_anything = False

    try:
        for chunk in llm.stream(messages):

            piece = getattr(chunk, "content", "") or ""

            if piece:
                produced_anything = True
                yield piece

    except Exception as e:
        raise OllamaServiceError(
            f"Could not generate an answer with Ollama model "
            f"'{OLLAMA_MODEL}'. Make sure Ollama is running and the "
            "model is installed."
        ) from e

    if not produced_anything:
        raise OllamaServiceError(
            "The model returned an empty response. The prompt may be too "
            "long for the model's context window, or the model may need "
            "to be reloaded."
        )
