'''
This is an Agentic RAG for activity transfer task, based on principle of vertical activity transform
activity (source) -> common activity -> activity (target)
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
from langchain_community.vectorstores import FAISS
#llm
from langchain_ollama import OllamaLLM
from langchain_community.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI # good for agents
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#agents
from langchain.tools import tool
from langchain.tools import Tool
#from langchain.agents import AgentType, initialize_agent # deprecated
from langchain.agents import AgentType, create_react_agent, AgentExecutor
#from langchain.memory import ConversationBufferMemory # being phased out
from langgraph.checkpoint.memory import MemorySaver
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import script_work.agent_database as agent_database
import script_work.agent_input as agent_input
import script_work.agent_query as agent_query
import script_work.agent_prompt as agent_prompt
from util import util_constants



# -----------------------
# TOOLS
# -----------------------
# @tool
# def retrieve_relevant_docs_goalstep(query: str):
#     """Retrieve the most relevant documents based on a user's query."""
#     return goalstep_retriever.invoke(query)

@tool
def check_goal_executability(query: str):
    """Checks if goal in the query is executable in the target scene"""
    # target scene 
    # source goal, and action sequence
    # 
    #if yes, pass the goal to target
    #if no, start find_common_activity tool

@tool
def find_common_activity(query: str):
    """Check common activity for both source and target space"""
    # source activity
    # see target space
    # sort out similar activities possible in target space
    # These activities are either stored or guessed by llm
    # if all failes, pluck out noun in activity
    # call check goal executability
    # if pass, return common activity
    # if fails for 3 times, announce failure to transfer

@tool
def narrow_down_activity(query: str):
    """Make input activity more specific and concrete, and check for executabliity"""
    #common activity -> target activity
    #check executability


@tool
def retrieve_relevant_docs_goalstep(query: str):
    """Retrieve the most relevant documents based on a user's query."""
    # return goalstep_retriever.invoke(query)

@tool
def retrieve_relevant_docs_spatial(query: str):
    """Retrieve the most relevant documents based on a user's query."""
    #return spatial_retriever.invoke(query)


# @tool
# def check_answer_relevance(question: str, answer: str) -> str:
#     """Check if the provided answer correctly and fully addresses the given question."""
#     prompt = f"Does this answer correctly and fully respond to the question?\n\n"
#     prompt += f"Question: {question}\nAnswer: {answer}\n"
#     prompt += "Reply with 'Yes' or 'No' and explain why."
#     return LLM_MODEL.invoke(prompt)

# @tool
# def enforce_lexical_constraints(answer: str) -> str:
#     """Ensure the answer only contains approved nouns and verbs."""
#     words = set(re.findall(r'\b\w+\b', answer.lower()))
#     invalid_words = words - ALLOWED_WORDS["nouns"] - ALLOWED_WORDS["verbs"]
#     if invalid_words:
#         return f"Invalid words found: {', '.join(invalid_words)}. Answer must use only allowed nouns and verbs."
#     return "Answer follows lexical constraints."

# -----------------------
# AGENT SETUP
# -----------------------
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
TOOLS = [
    check_goal_executability,
    find_common_activity,
    narrow_down_activity,
    retrieve_relevant_docs_goalstep
    ]
MEMORY = MemorySaver() #ConversationBufferMemory is deprecated


if __name__ == "__main__":
    # -----------------------
    # AGENT INPUT ARGUMENTS
    # -----------------------
    source_activity = ""
    target_scene_graph = ""
    tool_names =", ".join([t.name for t in TOOLS])

    # -----------------------
    # AGENT PROMPT
    # -----------------------
    AGENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are an activity transfer agent. You are"),
        ("system", "You are given a source activity. Source activity can be performed in source space without problem: {source_activity}."),
        ("system", "This time, source_activity should be performed in new space called target space. Info about target space is given as target_scene_graph: {target_scene_graph}."),
        ("system", "Available tools: {tools}. Use them wisely."),
        ("system", "Tool names: {tool_names}"),  # Required for React agents
        ("user", "{query}"),  # The user query should be directly included
        ("assistant", "{agent_scratchpad}")  # Required for React agents
    ])    


    # -----------------------
    # CREATE & RUN AGENT IN MAIN
    # -----------------------
    AGENT = create_react_agent(
        tools=TOOLS,  # Register tools
        llm=LLM_MODEL,
        prompt=AGENT_PROMPT
        #agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use react-based agent
        #verbose=True,  # Enable verbose output for debugging
        #checkpointer=MEMORY,
        #handle parsing error not built in for this function.
    )

    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS, 
        verbose=True, 
        handle_parsing_errors=True
    )    

    QUERY = ""

    response = AGENT_EXECUTOR.invoke({
        "query": QUERY,
        "target_activity": source_activity,
        "target_scene_graph": target_scene_graph,
        "tools": TOOLS,  # Pass tool objects
        "tool_names": ", ".join(TOOL_NAMES),  # Convert list to comma-separated string
        "agent_scratchpad": ""  # Let LangChain handle this dynamically
    })
    print(f"response {response}")