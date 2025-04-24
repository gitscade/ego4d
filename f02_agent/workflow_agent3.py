'''
This is an Agentic RAG for prediction of action sequence, based on given activity
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
# Path & API & Model
# -----------------------
data_path = util_constants.PATH_DATA
GOALSTEP_ANNOTATION_PATH = data_path + 'goalstep/'
SPATIAL_ANNOTATION_PATH = data_path + 'spatial/'
GOALSTEP_VECSTORE_PATH = GOALSTEP_ANNOTATION_PATH + 'goalstep_docarray_faiss'
SPATIAL_VECSTORE_PATH = SPATIAL_ANNOTATION_PATH + 'spatial_docarray_faiss'

logging.basicConfig(level=logging.ERROR)
load_dotenv() # load env variables
openai.api_key = os.getenv("OPENAI_API_KEY")
model1 = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini") #10x cheaper
parser_stroutput = StrOutputParser()

# -----------------------
# VIDEO LIST, VECSTORE, RETRIEVER
# -----------------------
ALLOWED_WORDS = {
    "nouns": {"apple", "banana", "computer", "data", "research"},
    "verbs": {"calculate", "analyze", "study", "process"}
}

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

def sequence_generation(input: str):

    input_dict = ast.literal_eval(input.strip())  # convert to python dict
    valid_json = json.dumps(input_dict, indent=4)  # read as JSON(wth "")
    input_json = json.loads(valid_json)
    query = input_json.get("query")
    target_activity = input_json.get("target_activity")
    target_scene_graph = input_json.get("target_scene_graph")

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
def sequence_validation(query: dict):
    prompt = f"You are an action sequence validator. You have to check three items. First, check that only entities from the target space is used for performing actions. Second, you need to see whether the actions are possible to be performed. Third, you need to check if the sequence of actions achieve the goal. When all three items pass, finalize the answer. Otherwise, re-try the action_sequence_generation tool for maximum of two times.\n"
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an action sequence validator"},
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    
    return response.choices[0].message.content


# -----------------------
# AGENT SETUP
# -----------------------
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
#LLM_MODEL_OLLAMA = OllamaLLM()
MEMORY = ConversationBufferWindowMemory(k=3)
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
        name = "action_sequence_generation_tool",
        func = sequence_generation,
        description = "Action sequence generation tool, which can break down the given activity into smaller actions. Additional information on the current target action, target environment is needed. Additional examples from other environments can also be used. Input: query(str), target_activity(str), . Output: action_sequence_steps(str)"
    ),        
    # Tool(
    #     name = "action_sequence_validation_tool",
    #     func = sequence_validation,
    #     description = "Input: query(str), action_sequence(str). Output: command to call action_sequence_generation_tool_obj again if validation fails. If validation passes, print out the input action_sequence(str)."
    # ),
    ]

def run_agent(target_video_idx=None, target_activity=""):
    
    if target_video_idx is None:
        target_video_idx = int(input("Input target index: "))
    if target_activity is "":
        target_activity = input("Input target activity: ")
    target_spatial_video = spatial_test_video_list[target_video_idx]
    target_scene_graph = agent_input.extract_spatial_context(target_spatial_video)
    tool_names =", ".join([t.name for t in TOOLS])

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=LLM_MODEL,
        prompt=agent_prompt.AGENT3_PROMPT
    )

    QUERY = "Give me a sequence of actions to fulfill the target_activity inside the environment of target_scene_graph"
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!
    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=MEMORY,
    )

    response = AGENT_EXECUTOR.invoke(
        {
            "query": QUERY,
            "target_activity": target_activity,
            "target_scene_graph": target_scene_graph,
            "tools": TOOLS,  # Pass tool objects
            "tool_names": tool_names,  # Convert list to comma-separated string
            "agent_scratchpad": ""  # Let LangChain handle this dynamically
        },
        config={"max_iterations": 3} 
        )
    return response


if __name__ == "__main__":

    target_video_idx = int(input("Input target index: "))
    target_activity = input("Input target activity: ")
    response = run_agent(target_video_idx, target_activity)