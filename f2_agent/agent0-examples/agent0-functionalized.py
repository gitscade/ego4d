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
    query = inputs[0]
    source_scene_graph = inputs[1]
    tools = inputs[2]
    tool_names = inputs[3]

    AGENT0_PROMPT = ChatPromptTemplate.from_messages(
        [
        ("system", 
        """You are an agent that answers queries using available tools. If you have gathered enough information, perform a step-by-step answering. First, explain your reasoning in a section labeled `Thought:`.Finally, give your answer in a section labeled `Final Answer:`.:
        
            Thought: Here is the final answer.
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain what tool invoke in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. source_scene_graph should be in JSON format for openAI:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: 
                {{
                "query": "{query}", 
                "source_scene_graph": {source_scene_graph} 
                }}
        """),

        ("system", "Available tools: {tools}. Here are the tools available for answering your question. Actively use retrieval tools to come up with plausible answer."),
        ("system", "Tool names: {tool_names}"),  # for React agents
        ("user", "{query}"),
        ("assistant", "{agent_scratchpad}")  # for React agents
        ]
        )

    # activity_prediction_message = [
    #     {"role": "system", "content": "You are a helpful assistant that uses scene graphs to gather information"},
    #     {"role": "user", "content": f"Here is the scene graph:\n{scene_graph_str}\n\nDescribe the structure of this scene?"}
    # ]
    activity_prediction_message = [
        {"role": "system", "content": "You are a helpful assistant that uses scene graphs to gather information"},
        {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n\nDescribe the structure of this scene?"}
    ]
    return AGENT0_PROMPT, activity_prediction_message

# -----------------------
# TOOL FUNCS = from packages
# -----------------------
# LLM TOOl ONLY FFEDS IN INPUT EXTERNALLY, NOT FROM AGENT to prevent formatting errors
def scene_explainer(activity_prediction_message):
    """Explain the layout of the scene"""
    try:
        # Parse string to dict — input is a string when used with LangChain agents
        # input_dict = json.loads(input)
        # query = input_dict.get("query")
        # source_scene_graph = input_dict.get("source_scene_graph")

        # Call OpenAI
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=activity_prediction_message,
            temperature=0.5
        )
        activity = response.choices[0].message.content.strip()
        return activity
    except Exception as e:
        return f"Tool Error: {str(e)}"

def get_agent0_tools():
    """
    return tools, tool_names
    """
    tools = [
    Tool(
        name = "scene_explanation_tool",
        func = lambda _: scene_explainer(activity_prediction_message),
        description = "Activity prediction tool, which can summarize the sequential multiple actions into a short single phrase of activity."
    ),
    ]

    tool_names =", ".join([t.name for t in tools])   
    return tools, tool_names 

# -----------------------
# AGENT FUNCS = from packages
# -----------------------
def run_agent0(tool0, tool_names, source_scene_graph):

    QUERY = "What type of room it this?."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!
    AGENT = create_react_agent(
        tools=tool0,
        llm=LLM_MODEL_AGENT,
        prompt=agent_prompt.AGENT0_PROMPT
    )    
    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=tool0, 
        verbose=True, 
        handle_parsingmory=MEMORY
    )

    response = AGENT_EXECUTOR.invoke(
        {
            "query": QUERY, 
            "source_scene_graph": source_scene_graph,
            "tools": tool0,  # Pass tool objects
            "tool_names": tool_names,  # Convert list to comma-separated string
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
    query = "What type of room it this?."    
    tools, tool_names = get_agent0_tools()
    inputs = [query, source_scene_graph_str, tools, tool_names]
    prompt, activity_prediction_message= get_agent0_message(inputs)
    # -----------------------
    # RUN AGENT
    # -----------------------
    response = run_agent0(tools, tool_names, source_scene_graph_str)
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