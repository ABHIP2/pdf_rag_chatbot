from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """
    Takes document chunks, embeds them using a
    HuggingFace model, stores in FAISS, returns the store.
    """
    # Free, local embedding model (no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS index from chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_embeddings_model():
    """Returns the embedding model (reused for queries)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )