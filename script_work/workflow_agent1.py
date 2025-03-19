'''
This is an Agentic RAG for recognition of activity based on input action sequence and spatial context
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

# Define LLM
llm = OpenAI(model_name="gpt-4", temperature=0)

# Define tool for agent1
tool1 = Tool(name="Tool1", func=lambda x: f"Processed by tool1: {x}", description="First tool")

#------------------------
#Tools
#------------------------
@tool
def retrieve_relevant_docs_goalstep(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    return goalstep_vector_store.invoke(query)

def retrieve_relevant_docs_spatial(query: str):
    """Retrieve the most relevant documents based on a user's query."""
    return spatial_retriever.invoke(query)


# -----------------------
# AGENT SETUP
# -----------------------
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1) #10x cheaper
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
TOOLS = [retrieve_relevant_docs_goalstep, retrieve_relevant_docs_spatial]
MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
AGENT_PROMPT = ChatPromptTemplate.from_messages(
    "You are an AI assistant agent that uses multiple tools to answer user query.\n"
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
    agent1_query = "What if the activity of the "
    print(run_agent("Start process"))