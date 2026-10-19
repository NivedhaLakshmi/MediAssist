"""The single embedding model shared by every vector store.

Loading `all-MiniLM-L6-v2` takes several seconds and holds the
weights in memory, so every caller imports the one instance from
here rather than building its own.
"""

from dotenv import load_dotenv

from services.offline_mode import prefer_offline

load_dotenv()

# Must run before the Hugging Face libraries are imported, because
# they read these variables at import time.
OFFLINE_REASON = prefer_offline()


from langchain_community.embeddings import (  # noqa: E402
    HuggingFaceEmbeddings,
)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)
