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
import f1_init.agent_init as agent_init
import f1_init.database_init as database_init
import f2_agent.agent_prompt as agent_prompt

"""
Final Chain of the Agent1,2,3
"""
    # -----------------------
    # MESSAGES / QUERIES FUNCS = from packages
    # -----------------------
    # -----------------------
    # TOOL FUNCS = from packages
    # -----------------------
    # -----------------------
    # AGENT FUNCS = from packages
    # -----------------------


if __name__ == "__main__":
    # -----------------------
    # API / LLM
    # -----------------------
    logging.basicConfig(level=logging.ERROR)
    load_dotenv()
    parser_stroutput = StrOutputParser()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = agent_init.LLM_MODEL_4MINI
    LLM_MODEL_AGENT = agent_init.LLM_MODEL_4MINI

    # -----------------------
    # FILES / FORMATTING
    # -----------------------
    source_video_idx = 1
    source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
    source_scene_graph = agent_init.extract_spatial_context(source_spatial_video)
    source_scene_graph_str = json.dumps(source_scene_graph, indent=2)

    target_video_idx = 1
    target_spatial_video = database_init.spatial_test_video_list[target_video_idx]
    target_scene_graph = agent_init.extract_spatial_context(target_spatial_video)
    target_scene_graph_str = json.dumps(target_scene_graph, indent=2)

    # -----------------------
    # MESSAGES / QUERIES
    # -----------------------

    # -----------------------
    # AGENT (FILES / MESSAGES / QUESRIES / TOOLS)
    # -----------------------
