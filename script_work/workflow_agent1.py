'''
This is an Agentic RAG for recognition of activity based on input action sequence and spatial context
input: source action sequence, source scene graph
context: additional vector dataset
output: source activity
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
import workflow_data

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
# CONFIGURE DATA
# -----------------------
ALLOWED_WORDS = {
    "nouns": {"apple", "banana", "computer", "data", "research"},
    "verbs": {"calculate", "analyze", "study", "process"}
}

# Load VIDEO LIST
spatial_test_video_list = workflow_data.spatial_test_video_list

# LOAD FAISS VECSTORE
goalstep_vector_store = workflow_data.goalstep_vector_store
spatial_vector_store = workflow_data.spatial_vector_store

# MAKE RETRIEVER
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


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
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
#LLM_MODEL_OLLAMA = OllamaLLM()
TOOLS = [
    goalstep_retriever_tool,
    spatial_retriever_tool,
    action_sequence_generation_tool,
    action_sequence_validation_tool
    ]
MEMORY = MemorySaver()


if __name__ == "__main__":
    # -----------------------
    # AGENT INPUT ARGUMENTS
    # -----------------------
    target_video_idx = int(input("Input target index: "))
    target_spatial_video = spatial_test_video_list[target_video_idx]
    target_scene_graph = agent_input.extract_spatial_context(target_spatial_video)
    target_activity = input("Input target activity: ")
    tool_names =", ".join([t.name for t in TOOLS])

    # -----------------------
    # AGENT PROMPT
    # -----------------------
    AGENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a ReACt agent that answers queries using tools. Always respond using this format:
         Thought: [Your reasoning]
         Action: [Tool name]
         Action Input: {{"query": "{{query}}", "target_activity": "{{target_activity}}", "target_scene_graph": "{{target_scene_graph}}" }}

         Example:
         Thought: I need to generate action sequence.
         Action: action_sequence_generation_tool
         Action Input: {{"query": "generate activity using target acitivity and target scene graph", "target_activity": "{{target_activity}}", "target_scene_graph": "{{target_scene_graph}}"}}
         """),
        ("system", "The user wants to perform a target activity: {target_activity}."),
        ("system", "The user is in a space described by this scene graph. Only use entities in this scene graph. Every state of each entity starts from here and can be changed during actions which effect the entity: {target_scene_graph}."),
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

    QUERY = "Give me a sequence of actions to fulfill the target_activity inside the environment of target_scene_graph"

    response = AGENT_EXECUTOR.invoke({
        "query": QUERY,
        "target_activity": target_activity,
        "target_scene_graph": target_scene_graph,
        "tools": TOOLS,  # Pass tool objects
        "tool_names": ", ".join(tool_names),  # Convert list to comma-separated string
        "agent_scratchpad": ""  # Let LangChain handle this dynamically
    })

    print(f"response {response}")





# # -----------------------
# # AGENT SETUP
# # -----------------------
# # LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1) #10x cheaper
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
# TOOLS = [retrieve_relevant_docs_goalstep, retrieve_relevant_docs_spatial]
# MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# AGENT_PROMPT = ChatPromptTemplate.from_messages(
#     "You are an AI assistant agent that uses multiple tools to answer user query.\n"
#     "Use the 'retrieve_relevant_docs' tool to fetch information before answering.\n"
#     "Whenever you generate an answer, use the following tools to verify it:\n"
#     "- 'check_answer_relevance' to ensure the answer actually answers the question.\n"
#     "- 'enforce_lexical_constraints' to verify that the answer only contains allowed words.\n"
#     "Only finalize an answer if it passes both checks.\n"
#     "User Query: {query}"
# )


# # -----------------------
# # RUN AGENT IN MAIN
# # -----------------------
# AGENT = initialize_agent(
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     tools= TOOLS,
#     llm=LLM_MODEL,
#     verbose=True,
#     memory=MEMORY,
#     handle_parsing_errors=True
# )

# def run_agent(input_text):
#     formatted_prompt = AGENT_PROMPT.format(query=input_text)
#     return AGENT.run(formatted_prompt)

# if __name__ == "__main__":
#     agent1_query = "What if the activity of the "
#     print(run_agent("Start process"))