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
import json
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

# -----------------------
# TOOL FUNCTION
# -----------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    context = goalstep_retriever.invoke(query)
    return f"User Query: {query}. similar goalstep examples: {context}" 

def spatial_information_retriver(query:dict):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    context = spatial_retriever.get_relevant_documents(query)
    return f"User Query: {query}. similar spatial examples: {context}"

# TODO2:
# fetching whole documents based on retrieval system.

def sequence_generation(input: str):
    # all json file must use double quotes
    input.replace("'", '"')
    params = json.loads(input)
    query = params.get("query")
    target_activity = params.get("target_activity")
    target_scene_graph = params.get("target_scene_graph")

    prompt = f"Here is the query: {query}. Here is the target_activity: {target_activity}. Here is the target_scene_graph: {target_scene_graph}"

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You generate action sequence based on input with three items. First input is query. Second input argument is target_activity, given in the prompt. Third input argument is target_scene_graph, also given in system prompt. Third input argument is context. It can either be given or left in black.\n ONLY use entities given by the target_scene_graph argument."},
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    
    action_sequence = response.choices[0].message.content.strip()

    return f"Thought: I need to generate the action sequence.\nAction: action_sequence_generation_tool\nAction Input: {json.dumps({'query': query, 'target_activity': target_activity, 'target_scene_graph': target_scene_graph})}\n{action_sequence}"


# TODO NEED FORMAT VALIDATION
def sequence_validation(query: dict, generated_action_sequence: str):
    prompt = f"You are an action sequence validator. You have to check three items. First, check that only entities from the target space is used for performing actions. Second, you need to see whether the actions are possible to be performed. Third, you need to check if the sequence of actions achieve the goal. When all three items pass, finalize the answer. Otherwise, re-try the action_sequence_generation tool for maximum of two times.\n"
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an action sequence validator"},
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    
    return response.choices[0].message.content


# -----------------------
# DEFINE TOOLS with TOOLFUNC
# -----------------------
goalstep_retriever_tool = Tool(
    name = "goalstep_retriever_tool",
    func = goalstep_information_retriever,
    description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
)

spatial_retriever_tool = Tool(
    name = "spatial_retriever_tool",
    func = spatial_information_retriver,
    description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
)

action_sequence_generation_tool = Tool(
    name = "action_sequence_generation_tool",
    func = sequence_generation,
    description = "Action sequence generation tool, which can break down the given activity into smaller actions. Additional information on the current target action, target environment is needed. Additional examples from other environments can also be used. Input: query(str), target_activity(str), . Output: action_sequence_steps(str)"
)

action_sequence_validation_tool = Tool(
    name = "action_sequence_validation_tool",
    func = sequence_validation,
    description = "Input: query(str), action_sequence(str). Output: command to call action_sequence_generation_tool_obj again if validation fails. If validation passes, print out the input action_sequence(str)."
)


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
    ]
#    action_sequence_validation_tool
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
        ("system", """You are an agent that answers queries using tools. If you have gathered enough information, respond with:
        
        Thought: I now have enough information to answer.
        Final Answer: [Your answer]
        
        Otherwise, use this format:
        Thought: [Your reasoning]
        Action: [Tool name]
        Action Input: {{"query": {query}, "target_activity":{target_activity}, "target_scene_graph": {target_scene_graph} }}
        """),        

        ("system", "The user wants to perform a target activity. The user is in a space described by this scene graph. Both target activity and scene graph is given in user message. Only use entities in this scene graph."),
        ("system", "Available tools: {tools}. Use them when necessary."),
        ("system", "Tool names: {tool_names}"), 
        ("system",  "user target activity: {target_activity}"),
        ("system",  "user target scene graph: {target_scene_graph}"),
        ("user", "user query: {query}"), 
        ("assistant", "{agent_scratchpad}") 
    ])
        # QUERY TO TEST
        # ("system", """You are a ReAct agent that answers queries using tools. Always respond using this format. When finalizing answer, just output final text.:
        #  Thought: [Your reasoning]
        #  Action: [Tool name]
        #  Action Input: {{"query": "{{query}}", "target_activity": "{{target_activity}}", "target_scene_graph": "{{target_scene_graph}}" }}

        #  Example:
        #  Thought: I need to generate action sequence.
        #  Action: action_sequence_generation_tool
        #  Action Input: {{"query": "generate activity using target acitivity and target scene graph", "target_activity": "{{target_activity}}", "target_scene_graph": "{{target_scene_graph}}"}}
        #  """),


    # -----------------------
    # CREATE & RUN AGENT IN MAIN
    # -----------------------
    AGENT = create_react_agent(
        tools=TOOLS,  # Register tools
        llm=LLM_MODEL,
        prompt=AGENT_PROMPT
    )
    #======DEPRECATED arguments===========
    #agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use react-based agent
    #verbose=True,  # Enable verbose output for debugging
    #checkpointer=MEMORY,
    #handle parsing error not built in for this function.

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


