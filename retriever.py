def get_retriever(vector_store, k=4):
    """
    Creates a retriever from the FAISS vector store.
    k = number of top chunks to retrieve per query.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever