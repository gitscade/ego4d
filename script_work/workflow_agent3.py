'''
This is an Agentic RAG for prediction of action sequence, based on given activity
'''
import sys
import os
import re
import pickle
import streamlit as st
import openai
import pandas as pd
import logging
from dotenv import load_dotenv
#vectorstore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
#llm
from langchain_ollama import OllamaLLM
from langchain.llms import OpenAI # good for single return task
from langchain_openai.chat_models import ChatOpenAI # good for agents
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#agents
from langchain.tools import tool
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import script_work.agent_database as agent_database
import script_work.agent_input as agent_input
import script_work.agent_query as agent_query
import script_work.agent_prompt as agent_prompt
from util import util_constants


# -----------------------
# CONFIGURATION
# -----------------------

ALLOWED_WORDS = {
    "nouns": {"apple", "banana", "computer", "data", "research"},
    "verbs": {"calculate", "analyze", "study", "process"}
}

# -----------------------
# Path & API & Model
# -----------------------
data_path = util_constants.PATH_DATA
GOALSTEP_ANNOTATION_PATH = data_path + 'goalstep/'
SPATIAL_ANNOTATION_PATH = data_path + 'spatial/'
GOALSTEP_VECSTORE_PATH = GOALSTEP_ANNOTATION_PATH + 'goalstep_docarray_faiss'
SPATIAL_VECSTORE_PATH = SPATIAL_ANNOTATION_PATH + 'spatial_docarray_faiss'

logging.basicConfig(level=logging.ERROR)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model1 = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini") #10x cheaper
parser_stroutput = StrOutputParser()

# -----------------------
# PREPROCESS DATA
# -----------------------
# EXTRACT video list
print(GOALSTEP_ANNOTATION_PATH)
print(SPATIAL_ANNOTATION_PATH)
goalstep_videos_list = agent_database.merge_json_video_list(GOALSTEP_ANNOTATION_PATH)
spatial_videos_list = agent_database.merge_json_video_list(SPATIAL_ANNOTATION_PATH)
print(f"goalstep vids: {len(goalstep_videos_list)} and spatial vids: {len(spatial_videos_list)}")

# EXCLUDE test videos
test_uid = [
    "dcd09fa4-afe2-4a0d-9703-83af2867ebd3", #make potato soap
    "46e07357-6946-4ff0-ba36-ae11840bdc39", #make tortila soap
    "026dac2d-2ab3-4f9c-9e1d-6198db4fb080", #prepare steak
    "2f46d1e6-2a85-4d46-b955-10c2eded661c", #make steak
    "14bcb17c-f70a-41d5-b10d-294388084dfc", #prepare garlic(peeling done)
    "487d752c-6e22-43e3-9c08-627bc2a6c6d4", #peel garlic
    "543e4c99-5d9f-407d-be75-c397d633fe56", #make sandwich
    "24ba7993-7fc8-4447-afd5-7ff6d548b11a", #prepare sandwich bread
    "e09a667f-04bc-49b5-8246-daf248a29174", #prepare coffee
    "b17ff269-ec2d-4ad8-88aa-b00b75921427", #prepare coffee and bread
    "58b2a4a4-b721-4753-bfc3-478cdb5bd1a8" #prepare tea and pie
]
goalstep_videos_list, goalstep_test_video_list = agent_database.exclude_test_video_list(goalstep_videos_list, test_uid)
spatial_videos_list, spatial_test_video_list = agent_database.exclude_test_video_list(spatial_videos_list, test_uid)
print(f"testuid excluded: goalstep vids: {len(goalstep_videos_list)} and spatial vids: {len(spatial_videos_list)}")
print(f"testuid list: goalstep vids: {len(goalstep_test_video_list)} and spatial vids: {len(spatial_test_video_list)}")

# MAKE docu list
goalstep_document_list = agent_database.make_goalstep_document_list(goalstep_videos_list)
spatial_document = agent_database.make_spatial_document_list(spatial_videos_list)
goalstep_test_document_list = agent_database.make_goalstep_document_list(goalstep_test_video_list)
spatial_test_document_list = agent_database.make_spatial_document_list(spatial_test_video_list)

print(f"MAKE_DOCU: goalstep_document_list: {len(goalstep_document_list)}")
print(f"MAKE_DOCU: spatial_document_list: {len(spatial_document)}")
print(f"MAKE_DOCU: goalstep_document_list: {len(goalstep_test_document_list)}")
print(f"MMAKE_DOCUAKE: spatial_document_list: {len(spatial_test_document_list)}")


# -----------------------
# MAKE/LOAD FAISS Vectorstore and retrievers
# -----------------------
embeddings = OpenAIEmbeddings()

if not os.path.exists(GOALSTEP_VECSTORE_PATH + '/index.faiss'):
    print(f"MAKE FAISS GOALSTEP: {GOALSTEP_VECSTORE_PATH}")
    goalstep_vector_store =  FAISS.from_documents(goalstep_document_list, embeddings)
    goalstep_vector_store.save_local(GOALSTEP_VECSTORE_PATH)
else:
    print(f"LOAD FAISS GOALSTEP: {GOALSTEP_VECSTORE_PATH}")

if not os.path.exists(SPATIAL_VECSTORE_PATH + '/index.faiss'):
    print(f"MAKE FAISS SPATIAL: {SPATIAL_VECSTORE_PATH}")
    spatial_vector_store = FAISS.from_documents(spatial_document, embeddings)
    spatial_vector_store.save_local(SPATIAL_VECSTORE_PATH)
else:
    print(f"LOAD FAISS SPATIAL: {SPATIAL_VECSTORE_PATH}")

# LOAD FAISS VECSTORE
goalstep_vector_store = FAISS.load_local(GOALSTEP_VECSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
spatial_vector_store = FAISS.load_local(SPATIAL_VECSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

# MAKE RETRIEVER
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# -----------------------
# TOOLS
# -----------------------
@tool
def retrieve_relevant_docs_goalstep(query: str):
    """Retrieve the most relevant documents based on a user's query."""
    return goalstep_retriever.invoke(query)


def retrieve_relevant_docs_spatial(query: str):
    """Retrieve the most relevant documents based on a user's query."""
    return spatial_retriever.invoke(query)

@tool
def retrieve_goalstep_docs(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    return goalstep_vector_store.invoke(query)

@tool
def retrieve_spatial_docs(query:str):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    return spatial_vector_store.invoke(query)

@tool
def check_answer_relevance(question: str, answer: str) -> str:
    """Check if the provided answer correctly and fully addresses the given question."""
    prompt = f"Does this answer correctly and fully respond to the question?\n\n"
    prompt += f"Question: {question}\nAnswer: {answer}\n"
    prompt += "Reply with 'Yes' or 'No' and explain why."
    return LLM_MODEL.invoke(prompt)

@tool
def enforce_lexical_constraints(answer: str) -> str:
    """Ensure the answer only contains approved nouns and verbs."""
    words = set(re.findall(r'\b\w+\b', answer.lower()))
    invalid_words = words - ALLOWED_WORDS["nouns"] - ALLOWED_WORDS["verbs"]

    if invalid_words:
        return f"Invalid words found: {', '.join(invalid_words)}. Answer must use only allowed nouns and verbs."
    return "Answer follows lexical constraints."



#check for right procedures
#check for right state changes(=right action)


# -----------------------
# AGENT SETUP
# -----------------------
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
LLM_MODEL = OpenAI(model_name="gpt-4", temperature=0)
TOOLS = [retrieve_relevant_docs_goalstep, retrieve_relevant_docs_spatial, check_answer_relevance, enforce_lexical_constraints]
MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
AGENT_PROMPT = ChatPromptTemplate.from_messages(
    "You are an AI assistant that retrieves and validates answers from documents.\n"
    "Use the 'retrieve_relevant_docs' tool to fetch information before answering.\n"
    "Whenever you generate an answer, use the following tools to verify it:\n"
    "- 'check_answer_relevance' to ensure the answer actually answers the question.\n"
    "- 'enforce_lexical_constraints' to verify that the answer only contains allowed words.\n"
    "Only finalize an answer if it passes both checks.\n"
    "User Query: {query}"
)

# -----------------------
# RUN AGENT IN MAIN
# -----------------------
AGENT = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools= TOOLS,
    llm=LLM_MODEL,
    verbose=True,
    memory=MEMORY,
    handle_parsing_errors=True
)

def run_agent(input_text):
    formatted_prompt = AGENT_PROMPT.format(query=input_text)
    return AGENT.run(formatted_prompt)


if __name__ == "__main__":
    print(run_agent("Start process"))


# # -----------------------
# # STREAMLIT UI
# # -----------------------
# def run_streamlit_app():
#     """Run the Streamlit UI in the same Python script."""
#     st.title("Agentic RAG with Streamlit")

#     # Input box for user query
#     query = st.text_area("Enter your query:", "")

#     # Button to trigger the agent
#     if st.button("Finalize Query"):
#         if query:
#             with st.spinner("Processing..."):
#                 response = agent.run(query)
#             st.subheader("Agent's Response:")
#             st.write(response)
#         else:
#             st.error("Please enter a query to continue.")

# if __name__ == "__main__":
    # run_streamlit_app()
