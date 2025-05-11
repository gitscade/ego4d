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
import f1_init.database_init as database_init
import f2_agent.agent_prompt as agent_prompt
from util import util_funcs

#------------------------
#prompt messages
#------------------------
def get_agent1a_message(inputs:list):
    """
    func: returns prompt and messages used for agent1a\n
    input: single list: [tools, sequence, scenegraph]\n
    return: AGENT1a_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """

    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]

    query = "summarize the input action sequence with a single verb and a single noun"
    tool_names =", ".join([t.name for t in tools])    

    AGENT1a_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful taxonomy summarizer that summarizes an action_sequence in input scene_graph, using tools. State your final answer following the format below:
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below.
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: 
                {{
                "query": "{query}", 
                "source_action_sequence": "{source_action_sequence}", 
                "source_scene_graph": "{source_scene_graph}" 
                }}
        """),
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),  # for React agents
        ("user", "{query}"),
        ("assistant", "{agent_scratchpad}")  # for React agents
        ])

    MESSAGE_ACTIVITY_PREDICTION = [
            {"role": "system", 
            "content": "You are a linguist that summarizes current user activity to a single verb and a single noun with a detailed thought process for the summary." }, 
            { "role": "user", "source_action_sequence": "{source_action_sequence}" },
            { "role": "user", "source_scene_graph": "{source_scene_graph}" },
        ]
    return AGENT1a_PROMPT, MESSAGE_ACTIVITY_PREDICTION

def get_agent1b_message(inputs:list):
    """
    func: returns prompt and messages used for agent1a\n
    input: single list: [tools, sequence, scenegraph, coreactivity]\n
    return: AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    source_core_activity = input[3]

    query = "make a 7-level taxonomy for the given core activity"
    tool_names =", ".join([t.name for t in tools])    

    AGENT1b_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful classifier that constructs a taxonomy of an activity in a scene. State your though process and final answer following the format below:
        
        Thought: Here is the answer from move_down_activity_tool.
        Final Answer: [Your answer]
            
        Otherwise, use this format:
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: {{"query": "{query}", "source_action_sequence": "{source_action_sequence}", "source_scene_graph": "{source_scene_graph}" }}
            """),

        ("system", "This is the user action sequence: {source_action_sequence}."),
        ("system", "Predicted activity must be able to be performed in this scene: {source_scene_graph}."),
        ("system", "Available tools: {tools}. Actively use retrieval tools to come up with plausible answer."),
        ("system", "Tool names: {tool_names}"),  # for React agents
        ("user", "{query}"),
        ("assistant", "{agent_scratchpad}")  # for React agents
        ])

    MESSAGE_TAXONOMY_CREATION = [

    ]

#------------------------
#Tool Funcs
#------------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    context = agent_init.goalstep_retriever.invoke(query)
    return f"User Query: {query}. similar goalstep examples: {context}" 

def spatial_information_retriver(query:dict):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    context = agent_init.spatial_retriever.invoke(query)
    return f"User Query: {query}. similar spatial examples: {context}"

def activity_prediction(MESSAGE_ACTIVITY_PREDICTION):
    """Predict an core summary of the user activity based on input"""
    try:
        #OPENAI
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model = agent_init.LLM_MODEL_AGENT,
            messages = MESSAGE_ACTIVITY_PREDICTION,
            temperature=0.5
        )      
        response = response.choices[0].message.content.strip()
        #OLLAMA
        # response = ollama.chat(
        #     model = LLM_MODEL_AGENT,
        #     messages= MESSAGE_ACTIVITY_PREDICTION,
        #     options={ 'temperature':0.5 }
        # )
        # response = response['message']['content']
        return response   

    except Exception as e:
        return f"Error: activity_prediction: {str(e)}"

# -----------------------
# Tool GET Funcs
# -----------------------
def get_agent1a_tools():
    """
    return tools
    """
    tools=[
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
        func = lambda _:activity_prediction(MESSAGE_ACTIVITY_PREDICTION),
        description = "Activity prediction tool, which can summarize the sequential multiple actions into a short single phrase of activity. Additional examples from other environments can also be used. Input: query(str), target_activity(str), . Output: action_sequence_steps(str)"
    ),
    ]
    return tools

def get_agent1b_tools():
    """
    return tools
    """
    tools = [
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
    return tools

# -----------------------
# Agent Function
# -----------------------
def run_agent_1a(input):
    """"
    func: run agent 1a with source video idx info
    input: [source_video_idx, tools]
    output: response
    """
    # Load input
    source_video_idx = input[0]
    TOOLS = input[1]
    TOOLNAMES =", ".join([t.name for t in TOOLS])    

    # Load & format documents
    if source_video_idx is None:
        source_video_idx = int(input("Input source index:"))
    source_goalstep_video = database_init.goalstep_test_video_list[source_video_idx]
    source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
    source_action_sequence = agent_init.extract_lower_goalstep_segments(source_goalstep_video)
    source_scene_graph = agent_init.extract_spatial_context(source_spatial_video)
    source_scene_graph = json.dumps(source_scene_graph)

    # AGENT
    QUERY = "Give me an phrase describing the activity of source_action_sequence."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!
    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_init.LLM_MODEL,
        prompt=agent_prompt.AGENT1a_PROMPT
    )    
    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=MEMORY
    )

    # RESPONSE
    response = AGENT_EXECUTOR.invoke(
        {
            "query": QUERY, 
            "source_scene_graph": source_scene_graph,
            "source_action_sequence": source_action_sequence,
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": ""
         },
        config={"max_iterations": 5}
    )
    return response

def run_agent_1b(input):
    """
    func: run agent 1a with source video idx info
    input: [source_video_idx, tools, source_activity]
    output: response
    """
    # Load input
    source_video_idx = input[0]
    TOOLS = input[1]
    TOOLNAMES =", ".join([t.name for t in TOOLS])
    source_activity= input[2]

    source_goalstep_video = database_init.goalstep_test_video_list[source_video_idx]
    source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
    source_action_sequence = agent_init.extract_lower_goalstep_segments(source_goalstep_video)
    source_action_sequence = util_funcs.convert_single_to_double_quotes_in_tuple(source_action_sequence)
    source_scene_graph = agent_init.extract_spatial_context(source_spatial_video)
    source_scene_graph = json.dumps(source_scene_graph)

    
    QUERY = "Categorically describe source activity in a very specific way."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_init.LLM_MODEL_AGENT,
        prompt=agent_prompt.AGENT1b_PROMPT
    )    

    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=MEMORY
    )

    response = AGENT_EXECUTOR.invoke(
        {
            "query": QUERY, 
            "source_scene_graph": source_scene_graph,
            "source_action_sequence": source_action_sequence,
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response


if __name__ == "__main__":

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
    # PREDICT CORE ACTIVITY
    # -----------------------
    tools_1a = get_agent1a_tools()
    input_1a_message = [source_video_idx, tools_1a]
    input_1a_agent = [source_video_idx, tools_1a]
    prompt, MESSAGE_ACTIVITY_PREDICTION= get_agent1a_message(input_1a_message)
    response_1a = run_agent_1a(input_1a)
    print(response_1a)

    # -----------------------
    # PREDICT FULL ACTIVITY TAXONOMY
    # -----------------------
    tools_1b = get_agent1b_tools()

    input1b_message = [tools_1b, sequence, scenegraph, coreactivity]
    input_1b_agent = [source_video_idx, tools_1b, source_activity]


    input_1b = [source_video_idx, tools_1b, response_1a]
    response_1b = run_agent_1b(input_1b)
    print(response_1b)




















#     TOOLS_1a = [
#     Tool(
#         name = "goalstep_retriever_tool",
#         func = goalstep_information_retriever,
#         description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
#     ),
#     Tool(
#         name = "spatial_retriever_tool",
#         func = spatial_information_retriver,
#         description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
#     ),
#     Tool(
#         name = "activity_prediction_tool",
#         func = activity_prediction,
#         description = "Activity prediction tool, which can summarize the sequential multiple actions into a short single phrase of activity. Additional examples from other environments can also be used. Input: query(str), target_activity(str), . Output: action_sequence_steps(str)"
#     ),
#     ]

# TOOLS_1b = [
#     Tool(
#         name = "goalstep_retriever_tool",
#         func = goalstep_information_retriever,
#         description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
#     ),
#     Tool(
#         name = "spatial_retriever_tool",
#         func = spatial_information_retriver,
#         description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
#     ),
#     Tool(
#         name = "move_down_activity_tool",
#         func = move_down_activity,
#         description = "This tool generates a spefic and deep hierarchical description of source activity"
#     ),
#     ]