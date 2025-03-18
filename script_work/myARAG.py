"""
Agentic Rag Example
"""

import os
import re
import pickle
import streamlit as st
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

# -----------------------
# CONFIGURATION
# -----------------------
OLLAMA_MODEL = "mistral"  # Change to "llama3", "gemma", etc.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENT_PATHS = ["docs/document1.pdf", "docs/document2.txt", "docs/data.csv"]
FAISS_INDEX_PATH = "faiss_index"

ALLOWED_WORDS = {
    "nouns": {"apple", "banana", "computer", "data", "research"},
    "verbs": {"calculate", "analyze", "study", "process"}
}

# -----------------------
# DOCUMENT LOADING & VECTORSTORE SETUP
# -----------------------
def load_documents(paths):
    """Load documents from PDFs, text, and CSV files."""
    docs = []
    for path in paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        elif path.endswith(".csv"):
            loader = CSVLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

documents = load_documents(DOCUMENT_PATHS)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load or create FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index...")
    with open(f"{FAISS_INDEX_PATH}/faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
else:
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(docs_chunks, embeddings)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(f"{FAISS_INDEX_PATH}/faiss_store.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -----------------------
# LLM (Ollama)
# -----------------------
llm = Ollama(model=OLLAMA_MODEL, temperature=0.7)

# -----------------------
# TOOLS
# -----------------------

@tool
def retrieve_relevant_docs(query: str):
    """Retrieve the most relevant documents based on a user's query."""
    return retriever.invoke(query)

@tool
def check_answer_relevance(question: str, answer: str) -> str:
    """Check if the provided answer correctly and fully addresses the given question."""
    prompt = f"Does this answer correctly and fully respond to the question?\n\n"
    prompt += f"Question: {question}\nAnswer: {answer}\n"
    prompt += "Reply with 'Yes' or 'No' and explain why."
    return llm.invoke(prompt)

@tool
def enforce_lexical_constraints(answer: str) -> str:
    """Ensure the answer only contains approved nouns and verbs."""
    words = set(re.findall(r'\b\w+\b', answer.lower()))
    invalid_words = words - ALLOWED_WORDS["nouns"] - ALLOWED_WORDS["verbs"]

    if invalid_words:
        return f"Invalid words found: {', '.join(invalid_words)}. Answer must use only allowed nouns and verbs."
    return "Answer follows lexical constraints."

# -----------------------
# AGENT SETUP
# -----------------------
tools = [retrieve_relevant_docs, check_answer_relevance, enforce_lexical_constraints]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_prompt = ChatPromptTemplate.from_template(
    "You are an AI assistant that retrieves and validates answers from documents.\n"
    "Use the 'retrieve_relevant_docs' tool to fetch information before answering.\n"
    "Whenever you generate an answer, use the following tools to verify it:\n"
    "- 'check_answer_relevance' to ensure the answer actually answers the question.\n"
    "- 'enforce_lexical_constraints' to verify that the answer only contains allowed words.\n"
    "Only finalize an answer if it passes both checks.\n"
    "User Query: {query}"
)

agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# -----------------------
# STREAMLIT UI
# -----------------------
def run_streamlit_app():
    """Run the Streamlit UI in the same Python script."""
    st.title("Agentic RAG with Streamlit")

    # Input box for user query
    query = st.text_area("Enter your query:", "")

    # Button to trigger the agent
    if st.button("Finalize Query"):
        if query:
            with st.spinner("Processing..."):
                response = agent.run(query)
            st.subheader("Agent's Response:")
            st.write(response)
        else:
            st.error("Please enter a query to continue.")


if __name__ == "__main__":
    run_streamlit_app()