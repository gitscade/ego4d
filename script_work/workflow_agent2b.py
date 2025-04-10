'''
func: specify deep activity to target level
input: common deep activity
output: target deep activity
'''
import sys
import os
import openai
import logging
import json
import ast
import argparse
from dotenv import load_dotenv
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
from langchain.agents import AgentType, create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
#from langgraph.checkpoint.memory import MemorySaver # Saves everyghing leading to overflow
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import script_work.agent_database as agent_database
import script_work.agent_input as agent_input
import script_work.agent_query as agent_query
import script_work.agent_prompt as agent_prompt
from Scripts.Utils.util import util_constants
import workflow_data

# -----------------------
# VIDEO LIST, VECSTORE, RETRIEVER
# -----------------------
# Load VIDEO LIST
goalstep_test_video_list = workflow_data.goalstep_test_video_list
spatial_test_video_list = workflow_data.spatial_test_video_list

# LOAD FAISS VECSTORE
goalstep_vector_store = workflow_data.goalstep_vector_store
spatial_vector_store = workflow_data.spatial_vector_store

# MAKE base:VectorStoreRetriever
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# -----------------------
# TOOL FUNCTION
# -----------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    context = goalstep_retriever.invoke(query)
    return f"User Query: {query}. similar goalstep examples: {context}" 

def spatial_information_retriver(query:dict):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    context = spatial_retriever.invoke(query)
    return f"User Query: {query}. similar spatial examples: {context}"

def check_executability(query:str):
    """Test if goal of deep activity can be met in current target_scene."""
    #LLM LOGIC TO SEE THIS

def move_down_activity(query:str):
    """Make deep activity more specific and concrete by lowering one level down its hierarchy"""
    #LLM LOGIC

# -----------------------
# AGENT SETUP
# -----------------------
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
#LLM_MODEL_OLLAMA = OllamaLLM()
MEMORY = ConversationBufferWindowMemory(k=3)
#MEMORY = MemorySaver() # Saves everyghing leading to overflow
TOOLS = [
    Tool(
        name = "goalstep_retriever_tool",
        func = goalstep_information_retriever,
        description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
    ),
    Tool(
        name = "spatial_retriever_tool",
        func = spatial_information_retriver,
        description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
    ),
    Tool(
        name = "executability_check_tool",
        func = check_executability,
        description = "Test if goal of deep activity can be met in current target_scene."
    ),
    Tool(
        name = "move_down_activity_tool",
        func = move_down_activity,
        description = "Make deep activity more specific and concrete by lowering one level down its hierarchy"
    )
    ]




if __name__ == "__main__":
    # -----------------------
    # ARGS / CREATE / EXECUTE / RES
    # -----------------------
    source_video_idx = int(input("Input source index:"))
    source_goalstep_video = goalstep_test_video_list[source_video_idx]
    source_spatial_video = spatial_test_video_list[source_video_idx]
    source_action_sequence = agent_input.extract_lower_goalstep_segments(source_goalstep_video)
    source_scene_graph = agent_input.extract_spatial_context(source_spatial_video)
    tool_names =", ".join([t.name for t in TOOLS])

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=LLM_MODEL,
        prompt=agent_prompt.AGENT1_PROMPT
    )

    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=MEMORY
    )

    QUERY = "Give me a sequence of actions to fulfill the target_activity inside the environment of target_scene_graph"
    response = AGENT_EXECUTOR.invoke(        
        {
            "query": QUERY,
            "source_action_sequence": source_action_sequence,
            "source_scene_graph": source_scene_graph,
            "tools": TOOLS,  # Pass tool objects
            "tool_names": ", ".join(tool_names),  # Convert list to comma-separated string
            "agent_scratchpad": ""  # Let LangChain handle this dynamically
        },
            max_iterations = 3
    )

    print(f"response {response}")
