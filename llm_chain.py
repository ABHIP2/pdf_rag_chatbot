from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024,
    )


def build_qa_chain(retriever):
    llm = get_llm()

    prompt = PromptTemplate.from_template("""You are a helpful assistant that answers questions
based strictly on the provided PDF context.

If the answer is not found in the context, say:
"I couldn't find that information in the uploaded PDF."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return both chain and retriever so we can fetch source pages
    return {"chain": chain, "retriever": retriever}


def ask_question(qa_bundle, question):
    chain = qa_bundle["chain"]
    retriever = qa_bundle["retriever"]

    # Run the answer chain
    answer = chain.invoke(question)

    # Fetch source docs for page numbers
    source_docs = retriever.invoke(question)
    pages_used = list(set([
        doc.metadata.get("page", 0) + 1
        for doc in source_docs
    ]))

    return answer, pages_used