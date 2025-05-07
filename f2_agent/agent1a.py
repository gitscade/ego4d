'''
func: predict deep activity for source scene, using source action sequece/scene graph/RAG examples
input: (source) action sequence, scene graph
output: source deep activity
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
import ollama
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
#from langgraph.checkpoint.memory import MemorySaver
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import f1_init.agent_init as agent_init
import f1_init.agent_input as agent_input
import f2_agent.agent_prompt as agent_prompt
import workflow_data
from util import util_funcs


# -----------------------
# API & LLM
# -----------------------
logging.basicConfig(level=logging.ERROR)
load_dotenv()
parser_stroutput = StrOutputParser()
openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL = agent_init.LLM_MODEL_LLAMA370b
LLM_MODEL_AGENT = agent_init.LLM_MODEL_LLAMA370b


# -----------------------
# VIDEO LIST, VECSTORE, RETRIEVER
# -----------------------
ALLOWED_WORDS = {
    "nouns": {"apple", "banana", "computer", "data", "research"},
    "verbs": {"calculate", "analyze", "study", "process"}
}

# Load VIDEO LIST (use text video list for testing)
goalstep_test_video_list = workflow_data.goalstep_test_video_list
spatial_test_video_list = workflow_data.spatial_test_video_list

# LOAD FAISS VECSTORE
goalstep_vector_store = workflow_data.goalstep_vector_store
spatial_vector_store = workflow_data.spatial_vector_store

# MAKE base:VectorStoreRetriever
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


#------------------------
#Tools
#------------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    context = goalstep_retriever.invoke(query)
    return f"User Query: {query}. similar goalstep examples: {context}" 

def spatial_information_retriver(query:dict):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    context = spatial_retriever.invoke(query)
    return f"User Query: {query}. similar spatial examples: {context}"

def activity_prediction(input: str):
    """Predict an activity of the user based on the input"""
    input_dict = ast.literal_eval(input.strip())  # convert to python dict
    valid_json = json.dumps(input_dict, indent=4)  # read as JSON(wth "")
    input_json = json.loads(valid_json)
    query = input_json.get("query")
    source_action_sequence = input_json.get("source_action_sequence")
    source_scene_graph = input_json.get("source_scene_graph")

    prompt = f"Here is the query: {query}. Here is the source_action_sequence: {source_action_sequence}. Here is the source_scene_graph: {source_scene_graph}"

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=LLM_MODEL_AGENT,
        messages=[
            {
            "role": "system", 
            "content": "You predict current user activity based on five input items. Activity MUST be given in one phrase inside a double quote. Answer format is as follows {{action in form of verb}} {{target in form of noun}}"
            }, 
            { "role": "user", "content": prompt}
                ],
        temperature=0.5
    )
    activity = response.choices[0].message.content.strip()

    response = ollama.chat(
        model = LLM_MODEL_AGENT,
        messages=[
            {
            "role": "system", 
            "content": "You predict current user activity based on five input items. Activity MUST be given in one phrase inside a double quote. Answer format is as follows {{action in form of verb}} {{target in form of noun}}"
            },
            { "role": "user", "content": prompt}
        ],
        options={
            'temperature':0.5
        }
    )
    activity = response['message']['content']


    return f"Thought: The activity is predicted.\nAction: activity_prediction_tool\nAction Input: {json.dumps({'query': query, 'source_action_sequence': source_action_sequence, 'source_scene_graph': source_scene_graph})}\n{activity}"

# TODO source activity must be given as input
def move_down_activity(input: str):
    """Make deep activity more specific and concrete by lowering one level down its hierarchy"""
    input_dict = ast.literal_eval(input.strip())  # convert to python dict
    valid_json = json.dumps(input_dict, indent=4)  # read as JSON(wth "")
    input_json = json.loads(valid_json)
    query = input_json.get("query")
    source_action_sequence = input_json.get("source_action_sequence")
    source_scene_graph = input_json.get("source_scene_graph")
    source_activity = input_json.get("source_activity")

    prompt = f"Here is the query: {query}. Here is the source_action_sequence: {source_action_sequence}. Here is the source_scene_graph: {source_scene_graph}. Here is the source_activity: {source_activity}"

    messages=[
        {
        "role": "system", 
        "content": "Return a serial of noun or words which is a specific and deep taxonomical description of source activity. For example, consider that ""cook steak"" is input activity, and sequence is about cooking steak. Construct a taxonomy of steak, that reflects the given action sequence. The taxonomy should take a form of: (cook)(steak)(meat)(salted)() and so on, with bracket enclosing noun description. The level of taxonomy should be at maxinum 5 levels deep."
        }, 
        { "role": "user", "content": prompt}
            ]
    
    json_messages = util_funcs.convert_single_to_double_quotes(messages)
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=LLM_MODEL_AGENT,
        messages = json_messages,
        temperature=0.5
    )
    categorized_activity = response.choices[0].message.content.strip()

    return f"Thought: Here is the categorized activity.\nAction: move_down_activity_tool\nAction Input: {json.dumps({'query': query, 'source_action_sequence': source_action_sequence, 'source_scene_graph': source_scene_graph})}\n{categorized_activity}"
    # return f"Thought: Here is the categorized activity.\n{categorized_activity}"



# -----------------------
# AGENT SETUP
# -----------------------

TOOLS_1a = [
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
        name = "activity_prediction_tool",
        func = activity_prediction,
        description = "Activity prediction tool, which can summarize the sequential multiple actions into a short single phrase of activity. Additional examples from other environments can also be used. Input: query(str), target_activity(str), . Output: action_sequence_steps(str)"
    ),
    ]

TOOLS_1b = [
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
        name = "move_down_activity_tool",
        func = move_down_activity,
        description = "This tool generates a spefic and deep hierarchical description of source activity"
    ),
    ]



def run_agent_1a(source_video_idx=None):

    if source_video_idx is None:
        source_video_idx = int(input("Input source index:"))
    source_goalstep_video = goalstep_test_video_list[source_video_idx]
    source_spatial_video = spatial_test_video_list[source_video_idx]
    source_action_sequence = agent_input.extract_lower_goalstep_segments(source_goalstep_video)
    source_scene_graph = agent_input.extract_spatial_context(source_spatial_video)
    tool_names =", ".join([t.name for t in TOOLS_1a])    

    AGENT = create_react_agent(
        tools=TOOLS_1a,
        llm=LLM_MODEL,
        prompt=agent_prompt.AGENT1a_PROMPT
    )    

    QUERY = "Give me an phrase describing the activity of source_action_sequence."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!
    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS_1a, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=MEMORY
    )
    response = AGENT_EXECUTOR.invoke(
        {
            "query": QUERY, 
            "source_scene_graph": source_scene_graph,
            "source_action_sequence": source_action_sequence,
            "tools": TOOLS_1a,  # Pass tool objects
            "tool_names": tool_names,  # Convert list to comma-separated string
            "agent_scratchpad": ""  # Let LangChain handle this dynamically
         },
        config={"max_iterations": 5}
    )
    return response


def run_agent_1b(source_video_idx=None, source_activity=""):
    """
    set input arguments
    set agent arguments
    create react agent
    create agent executor
    pass on responese
    """
    if source_video_idx is None:
        source_video_idx = int(input("Input source index:"))
    if source_activity is "":
        source_activity = input("Input source activity")
    source_goalstep_video = goalstep_test_video_list[source_video_idx]
    source_spatial_video = spatial_test_video_list[source_video_idx]
    source_action_sequence = agent_input.extract_lower_goalstep_segments(source_goalstep_video)
    source_action_sequence = util_funcs.convert_single_to_double_quotes_in_tuple(source_action_sequence)
    source_scene_graph = agent_input.extract_spatial_context(source_spatial_video)
    source_scene_graph = json.dumps(source_scene_graph)

    tool_names =", ".join([t.name for t in TOOLS_1b])    
    
    QUERY = "Categorically describe source activity in a very specific way."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS_1b,
        llm=LLM_MODEL_AGENT,
        prompt=agent_prompt.AGENT1b_PROMPT
    )    

    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS_1b, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=MEMORY
    )

    response = AGENT_EXECUTOR.invoke(
        {
            "query": QUERY, 
            "source_scene_graph": source_scene_graph,
            "source_action_sequence": source_action_sequence,
            "tools": TOOLS_1b,  # Pass tool objects
            "tool_names": tool_names,  # Convert list to comma-separated string
            "agent_scratchpad": ""  # Let LangChain handle this dynamically
         },
        config={"max_iterations": 5}
    )
    return response


if __name__ == "__main__":

    source_video_idx = int(input("Input source index:"))
    response_1a = run_agent_1a(source_video_idx)
    print(response_1a)
    source_activity = int(input("Input activity:"))
    response_1b = run_agent_1b(source_activity)
    print(response_1b)

# bef 0425 activity prediction tool prompt

#=========This is an openAI client=========
# client = openai.OpenAI()
# response = client.chat.completions.create(
#     model=LLM_MODEL_AGENT,
#     messages=[
#         {
#         "role": "system", 
#         "content": "You predict current user activity based on input with five items. The activity is a ONE PHRASE SUMMARY of the source_action_sequence in the input query. First input is query. Second input argument is source_action_sequence, given in the prompt. Third input argument is source_scene_graph, also given in system prompt. Fourth input argument is relevant_goalstep_information, sometimes given as None. Fifth input argument is relevant_scene_graph, sometimes given as None. Activity MUST be given in one phrase inside a double quote. Only one verb is allowed."
#         }, 
#         { "role": "user", "content": prompt}
#             ],
#     temperature=0.5
# )
# activity = response.choices[0].message.content.strip()