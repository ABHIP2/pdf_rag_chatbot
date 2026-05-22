import streamlit as st
from src.pdf_processor import load_and_split_pdf
from src.embeddings import create_vector_store
from src.retriever import get_retriever
from src.llm_chain import build_qa_chain, ask_question

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Chatbot")
st.caption("Powered by LangChain + Groq (llama3) + FAISS")

# ── Session state ─────────────────────────────────────────────
# Persists data across Streamlit reruns
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ── Sidebar: Upload PDF ───────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF to start chatting with it"
    )

    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("🔍 Processing PDF..."):
            # Step 1: Load and chunk
            chunks = load_and_split_pdf(uploaded_file)
            st.info(f"✅ Split into {len(chunks)} chunks")

            # Step 2: Embed and store
            with st.spinner("🧠 Building vector index..."):
                vector_store = create_vector_store(chunks)

            # Step 3: Create retriever + chain
            retriever = get_retriever(vector_store, k=4)
            st.session_state.qa_chain = build_qa_chain(retriever)
            st.session_state.pdf_processed = True
            st.session_state.chat_history = []  # Reset on new upload

        st.success(f"✅ **{uploaded_file.name}** is ready!")
        st.caption(f"Total chunks: {len(chunks)}")

    # Reset button
    if st.session_state.pdf_processed:
        if st.button("🔄 Upload a different PDF"):
            st.session_state.pdf_processed = False
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.rerun()

# ── Main chat area ────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("pages"):
                st.caption(f"📖 Source pages: {msg['pages']}")

    # Chat input
    if user_input := st.chat_input("Ask anything about the PDF..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, pages = ask_question(
                    st.session_state.qa_chain,
                    user_input
                )
            st.markdown(answer)
            if pages:
                st.caption(f"📖 Source pages: {pages}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "pages": pages
        })