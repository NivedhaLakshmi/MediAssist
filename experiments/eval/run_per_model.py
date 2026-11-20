"""Run the answer and hallucination scenarios once per answering model.

Each model gets the same prompts the application builds, at temperature 0
and the app's 8192-token context. Results are saved per model, for
example results/responses_llama3_2.json and
results/hallucination_qwen2_5_7b-instruct.json.

    python experiments/eval/run_per_model.py
"""

import re

import common  # noqa: F401  (sets the project root and working directory)
from langchain_ollama import ChatOllama

import eval_hallucination
import eval_responses
from services import llm_service

MODELS = ["llama3.2", "llama3", "qwen2.5:7b-instruct"]


def main():
    for model in MODELS:
        print("########", model, flush=True)
        llm_service.llm = ChatOllama(model=model, temperature=0, num_ctx=8192)
        tag = re.sub(r"[:.]", "_", model)
        eval_responses.main(save_as="responses_" + tag)
        eval_hallucination.main(save_as="hallucination_" + tag)


if __name__ == "__main__":
    main()
