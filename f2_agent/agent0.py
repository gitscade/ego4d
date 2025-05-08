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

1. API / LLM
2. FILES / FORMATTING
3. PROMPTS / MESSAGES
4. TOOL FUNCS
5. AGENT FUNCS
6. MAIN
"""


# -----------------------
# 1. API / LLM
# -----------------------
logging.basicConfig(level=logging.ERROR)
load_dotenv()
parser_stroutput = StrOutputParser()
openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL = agent_init.LLM_MODEL_4MINI
LLM_MODEL_AGENT = agent_init.LLM_MODEL_4MINI

# -----------------------
# 2. FILES / FORMATTING
# -----------------------
source_video_idx = 1
source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
source_scene_graph = agent_init.extract_spatial_context(source_spatial_video)
scene_graph_str = json.dumps(source_scene_graph, indent=2)

# -----------------------
# PROMPTS / MESSAGES
# -----------------------
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


activity_prediction_message = [
    {"role": "system", "content": "You are a helpful assistant that uses scene graphs to gather information"},
    {"role": "user", "content": f"Here is the scene graph:\n{scene_graph_str}\n\nDescribe the structure of this scene?"}
]

# -----------------------
# 4. TOOL FUNCS
# -----------------------
def scene_explainer(input: str):
    """Explain the layout of the scene"""
    try:
        # Parse string to dict — input is a string when used with LangChain agents
        input_dict = json.loads(input)
        query = input_dict.get("query")
        source_scene_graph = input_dict.get("source_scene_graph")

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

tool0 = [
Tool(
    name = "scene_explanation_tool",
    func = scene_explainer,
    description = "Activity prediction tool, which can summarize the sequential multiple actions into a short single phrase of activity."
),
]

# -----------------------
# 5. AGENT FUNCS
# -----------------------
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

source_scene_graph = json.dumps(source_scene_graph)
tool_names =", ".join([t.name for t in tool0])   
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

# -----------------------
# 6. MAIN
# -----------------------
if __name__ == "__main__":
    print("none")