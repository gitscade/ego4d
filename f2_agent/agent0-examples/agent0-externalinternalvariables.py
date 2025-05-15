#last check 250508
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
"""
Follow this flow to prevent formatting error.
NOT defining files used in queries or messages will require agent to pass on inputs using strict formatting.
PASSING input variables through agent will result in internal formatting by AGENT API.
INTERNAL FORMATTINGs by AGENT API can lead to incompatible variable format for the LLM API.

To prevent unforseen problems caused by inherent formatting, pre-format files first, then define queries & messages. 
"""
# -----------------------
# MESSAGES / QUERIES FUNCS = from packages
# -----------------------
def get_agent0_message(inputs: list):
    """
    func: return PROMPT / MESSAGES for agent1a
    input: [video_idx, tool, tool_names]
    return : AGENT0_PROMPT, SCENE_EXPLAINER_MESSAGE
    """
    video_idx = inputs[0]
    tools = inputs[1]
    tool_names = inputs[2]

    # FILES
    query = "what is the user making?"
    source_video_idx = video_idx

    source_goalstep_video = database_init.goalstep_test_video_list[source_video_idx]
    source_action_sequence = agent_init.extract_lower_goalstep_segments(source_goalstep_video)
    source_action_sequence = json.dumps(source_action_sequence)

    source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
    source_scene_graph = agent_init.extract_spatial_context(source_spatial_video)
    source_scene_graph = json.dumps(source_scene_graph, indent=2)

    AGENT0_PROMPT = ChatPromptTemplate.from_messages(
        [
        ("system", 
        """You are an agent that follows the ReAct reasoning format to answer queries using available tools. 
            
        When answering, always use this format below:
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: [
                {{"query": "{query}"}}
            ]

        Do not give final answer until you are done with all Actions. Give final anwer in this form:
            Thought: Here is the final answer.
            Final Answer: [Your answer]
        """),

        ("system", "Available tools: {tools}. Here are the tools available for answering your question. Actively use retrieval tools to come up with plausible answer."),
        ("system", "Tool names: {tool_names}"),  # for React agents
        ("user", "{query}"),
        ("assistant", "{agent_scratchpad}")  # for React agents
        ]
        )

        # Using Json format files like scene_graph give too much trouble! Refrain from using this internally
        #     Action Input: [
        #         {{"query": "{query}"}}, 
        #         {{"scene_graph": "{source_scene_graph}"}}
        #     ]
        # When not giving final answer always use this format below:
        #     Thought: [Your reasoning]
        #     Action: [Tool name]
        #     Action Input: ["query": "{query}", "source_scene_graph": {source_scene_graph}]
        # """),    
    
    TOOL_MESSAGE_SCENE_EXPLAINER = [
        {"role": "system", "content": "You are a helpful assistant that uses scene graphs to gather information"},
        {"role": "user", "content": f"Here is the action sequence:\n{source_action_sequence}\n\nDescribe the action sequence?"},
        {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n\nDescribe the structure of this scene?"}
    ]      
    return AGENT0_PROMPT, TOOL_MESSAGE_SCENE_EXPLAINER

# -----------------------
# TOOL FUNCS = from packages
# -----------------------
# LLM TOOl ONLY FFEDS IN INPUT EXTERNALLY, NOT FROM AGENT to prevent formatting errors
def scene_explainer(agent_input):
    """Explain the layout of the scene"""
    
    agent_input = json.loads(agent_input)
    query = agent_input[0]["query"]
    query = [{"role": "user", "content": query}]
    message = query + TOOL_MESSAGE_SCENE_EXPLAINER
    my_temperature = 0.5
    try:
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = message,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= message,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    

    except Exception as e:
        return f"Tool Error: {str(e)}"

def get_agent0_tools():
    """
    return tools, tool_names
    """
    tools = [
    Tool(
        name = "scene_explanation_tool",
        func = scene_explainer,
        description = "describe activity of the scene using the entities in the scene graph."
    ),
    ]

    tool_names =", ".join([t.name for t in tools])   
    return tools, tool_names 

# -----------------------
# AGENT FUNCS = from packages
# -----------------------
def run_agent0(input, agent_llm_chat):
    """
    func: run agent0
    input: [video_idx, tool, AGENT0_PROMPT]
    """
    video_idx = input[0]
    TOOLS = input[1]
    AGENT0_PROMPT = input[2]
    TOOLNAMES = ", ".join([t.name for t in TOOLS])    
    # FILES
    source_video_idx = video_idx
    source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
    source_scene_graph = agent_init.extract_spatial_context(source_spatial_video)
    source_scene_graph = json.dumps(source_scene_graph, indent=2)

    # AGENT
    QUERY = "What is the user making?."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!
    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_llm_chat,
        prompt= AGENT0_PROMPT
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
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": ""  # Let LangChain handle this dynamically
            },
        config={"max_iterations": 5}
    )
    return response

# -----------------------
# 6. MAIN
# -----------------------
if __name__ == "__main__":
    # -----------------------
    # API / LLM
    # -----------------------
    logging.basicConfig(level=logging.ERROR)
    load_dotenv()
    AGENT_LLM_API, AGENT_LLM_STR, AGENT_LLM_CHAT = agent_init.SET_LLMS("ollama", "gemma3:27b")
    TOOL_LLM_API, TOOL_LLM_STR, TOOL_LLM_CHAT = agent_init.SET_LLMS("ollama", "gemma3:27b")    

    # -----------------------
    # MESSAGES / QUERIES
    # -----------------------
    tools, tool_names = get_agent0_tools()
    video_idx = 1
    inputs = [video_idx, tools, tool_names]
    AGENT0_PROMPT, TOOL_MESSAGE_SCENE_EXPLAINER= get_agent0_message(inputs)
    # -----------------------
    # RUN AGENT
    # -----------------------
    input_agent0 = [video_idx, tools, AGENT0_PROMPT]
    response = run_agent0(input_agent0, AGENT_LLM_CHAT)
    print(response)






#  QUERY / SOURCE_SCENE_GRAPH / SOURCE_SCENE_GRAPH_STR

#   query =  "What type of room it this?.", 
#   source_scene_graph = [
#     {"object_id": 3, "object_name": "soup", "init_status": {"status": "uncooked", "container": null}}, 
#     {"object_id": 14, "object_name": "counter top", "init_status": {"status": "default", "container": null}}, 
#     {"object_id": 4, "object_name": "oil", "init_status": {"status": "default", "container": 14}}, 
#     {"object_id": 5, "object_name": "stirrer", "init_status": {"status": "default", "container": 14}}, 
#     {"object_id": 6, "object_name": "fridge", "init_status": {"status": "default", "container": null}}, 
#     {"object_id": 7, "object_name": "tortilla", "init_status": {"status": "wrapped", "container": 6}}, 
#     {"object_id": 8, "object_name": "table", "init_status": {"status": "default", "container": null}}, 
#     {"object_id": 9, "object_name": "a bottle of water", "init_status": {"status": "contain water", "container": null}}, 
#     {"object_id": 19, "object_name": "stovetop", "init_status": {"status": "default", "container": null}}, 
#     {"object_id": 10, "object_name": "pepper spice", "init_status": {"status": "default", "container": 19}}, 
#     {"object_id": 11, "object_name": "herbs", "init_status": {"status": "default", "container": 19}}, 
#     {"object_id": 12, "object_name": "chilli flakes", "init_status": {"status": "default", "container": 19}}, 
#     {"object_id": 13, "object_name": "spice", "init_status": {"status": "default", "container": 19}}, 
#     {"object_id": 15, "object_name": "sweet corn", "init_status": {"status": "default", "container": 19}}, 
#     {"object_id": 16, "object_name": "sieve", "init_status": {"status": "default", "container": null}}, 
#     {"object_id": 17, "object_name": "sink", "init_status": {"status": "default", "container": null}}, 
#     {"object_id": 18, "object_name": "peeler", "init_status": {"status": "unwashed", "container": 17}}, 
#     {"object_id": 20, "object_name": "skillet", "init_status": {"status": "default", "container": 19}}, 
#     {"object_id": 21, "object_name": "player", "init_status": {"status": null}}, 
#     {"object_id": 22, "object_name": "minced meat", "init_status": {"status": "unpacked", "container": null}}, 
#     {"object_id": 23, "object_name": "knife", "init_status": {"status": "default", "container": null}}
#   ] 
#   source_scene_graph_str ="source_scene_graph": [
#     {
#       "object_id": 3,
#       "object_name": "soup",
#       "init_status": {
#         "status": "uncooked",
#         "container": null
#       }
#     },
#     {
#       "object_id": 14,
#       "object_name": "counter top",
#       "init_status": {
#         "status": "default",
#         "container": null
#       }
#     },
#     {
#       "object_id": 4,
#       "object_name": "oil",
#       "init_status": {
#         "status": "default",
#         "container": 14
#       }
#     },
#     {
#       "object_id": 5,
#       "object_name": "stirrer",
#       "init_status": {
#         "status": "default",
#         "container": 14
#       }
#     },
#     {
#       "object_id": 6,
#       "object_name": "fridge",
#       "init_status": {
#         "status": "default",
#         "container": null
#       }
#     },
#     {
#       "object_id": 7,
#       "object_name": "tortilla",
#       "init_status": {
#         "status": "wrapped",
#         "container": 6
#       }
#     },
#     {
#       "object_id": 8,
#       "object_name": "table",
#       "init_status": {
#         "status": "default",
#         "container": null
#       }
#     },
#     {
#       "object_id": 9,
#       "object_name": "a bottle of water",
#       "init_status": {
#         "status": "contain water",
#         "container": null
#       }
#     },
#     {
#       "object_id": 19,
#       "object_name": "stovetop",
#       "init_status": {
#         "status": "default",
#         "container": null
#       }
#     },
#     {
#       "object_id": 10,
#       "object_name": "pepper spice",
#       "init_status": {
#         "status": "default",
#         "container": 19
#       }
#     },
#     {
#       "object_id": 11,
#       "object_name": "herbs",
#       "init_status": {
#         "status": "default",
#         "container": 19
#       }
#     },
#     {
#       "object_id": 12,
#       "object_name": "chilli flakes",
#       "init_status": {
#         "status": "default",
#         "container": 19
#       }
#     },
#     {
#       "object_id": 13,
#       "object_name": "spice",
#       "init_status": {
#         "status": "default",
#         "container": 19
#       }
#     },
#     {
#       "object_id": 15,
#       "object_name": "sweet corn",
#       "init_status": {
#         "status": "default",
#         "container": 19
#       }
#     },
#     {
#       "object_id": 16,
#       "object_name": "sieve",
#       "init_status": {
#         "status": "default",
#         "container": null
#       }
#     },
#     {
#       "object_id": 17,
#       "object_name": "sink",
#       "init_status": {
#         "status": "default",
#         "container": null
#       }
#     },
#     {
#       "object_id": 18,
#       "object_name": "peeler",
#       "init_status": {
#         "status": "unwashed",
#         "container": 17
#       }
#     },
#     {
#       "object_id": 20,
#       "object_name": "skillet",
#       "init_status": {
#         "status": "default",
#         "container": 19
#       }
#     },
#     {
#       "object_id": 21,
#       "object_name": "player",
#       "init_status": {
#         "status": null
#       }
#     },
#     {
#       "object_id": 22,
#       "object_name": "minced meat",
#       "init_status": {
#         "status": "unpacked",
#         "container": null
#       }
#     },
#     {
#       "object_id": 23,
#       "object_name": "knife",
#       "init_status": {
#         "status": "default",
#         "container": null
#       }
#     }
#   ]