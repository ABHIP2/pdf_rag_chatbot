# Chat with PDF

A RAG-based PDF chatbot built with LangChain, Groq, FAISS, and Streamlit.

## Features
- Upload any PDF and chat with it
- Powered by Groq's llama3 (ultra-fast inference)
- HuggingFace embeddings (runs locally, no extra API key)
- FAISS vector store for semantic search
- Built with Streamlit

## Tech Stack
- LangChain (LCEL chains)
- Groq API (llama-3.3-70b-versatile)  # Use your API Key 
- FAISS (vector store)
- HuggingFace (all-MiniLM-L6-v2 embeddings)
- Streamlit (UI)

## Setup

1. Clone the repo
   git clone https://github.com/yourusername/chat-with-pdf.git
   cd chat-with-pdf

2. Create and activate virtual environment
   python -m venv chatenv
   chatenv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Add your Groq API key
   **Create a .env file**
   GROQ_API_KEY=your_key_here

5. Run the app
   python -m streamlit run app.py


## Get Groq API Key
Free at https://console.groq.com
Create Your Own API Key and Use it In the env file
