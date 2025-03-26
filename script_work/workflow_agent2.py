'''
Agent for activity(source)->activity(target) matching
activity (source) -> common activity -> activity (target)

input: activity(source)
context: 
output: activity(source)
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













    # def run_agent(target_activity, target_scene_graph, AGENT, AGENT_PROMPT):
    #     formatted_prompt = AGENT_PROMPT.format(target_activity=target_activity, target_scene_graph=target_scene_graph)
    #     return AGENT.run(formatted_prompt)

    #print(run_agent(target_activity, target_scene_graph, AGENT, AGENT_PROMPT))

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
