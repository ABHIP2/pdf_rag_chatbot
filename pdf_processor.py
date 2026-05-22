from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def load_and_split_pdf(uploaded_file):
    """
    Takes a Streamlit UploadedFile object,
    saves it temporarily, loads with PyPDFLoader,
    splits into chunks, and returns the chunks.
    """
    # Save uploaded file to a temp path (PyPDFLoader needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load PDF pages
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # characters per chunk
        chunk_overlap=200,    # overlap to preserve context
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Clean up temp file
    os.unlink(tmp_path)

    return chunks