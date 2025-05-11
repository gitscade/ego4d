'''
func: move up source deep activity level so that resulting common deep activity is executable in source and target scene
input: source deep activity
output: comon deep activity
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
import f1_init.agent_init as agent_init
import f1_init.database_init as database_init
import f2_agent.agent_prompt as agent_prompt


#------------------------
#prompt messages
#------------------------
def get_agent2a_message(inputlist:list):
    """
    func: returns prompt and messages used for agent2a\n
    input: single list: [tools, sequence, scenegraph, target_scenegraph, source_activity_taxonomy, target_scene_graph]\n
    return: AGENT2a_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputlist[0]
    source_action_sequence = inputlist[1]
    source_scene_graph = inputlist[2]
    source_activity_taxonomy = inputlist[3]
    target_scene_graph = inputlist[4]
    query = "Construct a common taxonomy that is applicable for both source and target scene graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT2a_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful taxonomy tester that examines if each leval of the taxonomy is executable in target scene, using tools. State your final taxonomy in a section labeled 'Final Answer:'.:
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}, "source_action_sequence": {source_action_sequence}, "source_scene_graph": {source_scene_graph}, "source_activity_taxonomy": {source_activity_taxonomy}, "target_scene_graph": {target_scene_graph}]
        """),

        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )

    MESSAGE_COMMON_TAXONOMY_PREDICTION = [
            {"role": "system", "content": "You are a linguist that checks if each level in taxonomy can be achieved in the target scene. For each level in taxonomy, check if noun property can be achieved in the target scene. If this is possible, just leave the value for the noun. If noun property is not achievable, replace the original noun with word 'impossible'. Print out the final taxonomy as output."}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"}
        ]
    return AGENT2a_PROMPT, MESSAGE_COMMON_TAXONOMY_PREDICTION

def get_agent2b_message(inputs:list):
    """
    func: returns prompt and messages used for agent2b\n
    input: single list: [tools, sequence, scenegraph, target_scenegraph, source_taxonomy, common_taxonomy]\n
    return: AGENT2b_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]
    source_activity_taxonomy = inputs[4]
    common_activity_taxonomy = inputs[5]

    query = "summarize the input action sequence with a single verb and a single noun"
    tool_names =", ".join([t.name for t in tools])    


    AGENT2b_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful taxonomy summarizer that summarizes an action_sequence in input scene_graph, using tools. State your final answer in a section labeled 'Final Answer:'.:
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}, "source_action_sequence": {source_action_sequence}, "source_scene_graph": {source_scene_graph}, "source_activity_taxonomy": {source_activity_taxonomy}, "target_scene_graph": {target_scene_graph}, "common_activity_taxonomy": {common_activity_taxonomy}]
        """),

        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "{query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )

    MESSAGE_TARGET_TAXONOMY_PREDICTION = [
            {"role": "system", "content": "You are a linguist that summarizes current user activity to a single verb and a single noun"}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
            {"role": "user", "content": f"Here is the common activity taxonomy:\n{common_activity_taxonomy}\n"},            
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"}            
        ]
    return AGENT2b_PROMPT, MESSAGE_TARGET_TAXONOMY_PREDICTION

# -----------------------
# TOOL FUNCTION
# -----------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    context = agent_init.goalstep_retriever.invoke(query)
    return f"User Query: {query}. similar goalstep examples: {context}" 

def spatial_information_retriver(query:dict):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    context = agent_init.spatial_retriever.invoke(query)
    return f"User Query: {query}. similar spatial examples: {context}"

def check_taxonomy(MESSAGE_COMMON_TAXONOMY_PREDICTION):
    """Test if goal of deep activity can be met in current target_scene."""

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=agent_init.LLM_MODEL_4MINI,
        messages= MESSAGE_COMMON_TAXONOMY_PREDICTION,
        temperature=0.5
    )
    executability = response.choices[0].message.content.strip()
    return f"Thought: Executability.\n{executability}"

def move_up_activity(query:str):
    """Make deep activity more abstract by moving one level up its hierarchy"""
    #LLM LOGIC TO RULE OUT MOST SPECIFIC DECORATOR?

def move_down_activity(query:str):
    """Make deep activity more specific and concrete by lowering one level down its hierarchy"""
    #LLM LOGIC


# -----------------------
# AGENT SETUP
# -----------------------
MEMORY = ConversationBufferWindowMemory(k=3)
#MEMORY = MemorySaver() # Saves everyghing leading to overflow


def get_agent2a_tools():
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
            name = "executability_check_tool",
            func = lambda _: check_taxonomy(MESSAGE_COMMON_TAXONOMY_PREDICTION),
            description = "Test if goal of deep activity can be met in current target_scene."
        ),
        Tool(
            name = "move_up_activity_tool",
            func = move_up_activity,
            description = "Make deep activity more abstract by moving one level up its hierarchy"
        ),    
    ]
    return tools

def get_agent2b_tools():
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
    return tools




def run_agent_2a(source_video_idx=None, target_video_idx=None, source_activity=""):
    """
    func: source_taxonomy -> common_taxonomy
    input: 
    (source_taxonomy)
    (source_sequence, source_scenegraph)
    (target_scenegraph)
    output: 
    (common_taxonomy)
    """
    common_taxonomy = []


    return common_taxonomy


def run_agent_2b(source_video_idx=None, target_video_idx=None, common_activity=""):
    """
    func: common_taxonomy -> target_taxonomy
    input: 
    (common_taxonomy)
    (source_sequence, source_scenegraph)
    (target_scenegraph)
    output: 
    (target_taxonomy)
    """
    target_taxonomy = []

    return target_taxonomy

if __name__ == "__main__":

    # -----------------------
    # FILES / FORMATTING
    # -----------------------
    source_video_idx = 1
    target_video_idx = 2
    
    source_goalstep_video = database_init.goalstep_test_video_list[source_video_idx]
    source_action_sequence = agent_init.extract_lower_goalstep_segments(source_goalstep_video)
    source_action_sequence = json.dumps(source_action_sequence)
    source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
    source_scene_graph = agent_init.extract_spatial_context(source_spatial_video)
    source_scene_graph = json.dumps(source_scene_graph, indent=2)

    target_spatial_video = database_init.spatial_test_video_list[target_video_idx]
    target_scene_graph = agent_init.extract_spatial_context(target_spatial_video)
    target_scene_graph = json.dumps(target_scene_graph, indent=2)
    
    # -----------------------
    # PREDICT COMMON ACTIVITY
    # -----------------------
    """
    func: returns prompt and messages used for agent2a\n
    input: single list: [tools, sequence, scenegraph, target_scenegraph, source_activity_taxonomy, target_scene_graph]\n
    return: AGENT2a_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools_2a = get_agent2a_tools()



    # -----------------------
    # PREDICT FULL ACTIVITY TAXONOMY
    # -----------------------
    tools_2b = get_agent2b_tools()






#openai client
    # client = openai.OpenAI()
    # response = client.chat.completions.create(
    #     model=LLM_MODEL_AGENT
    #     messages=[
    #         {
    #         "role": "system", 
    #         "content": "Return a serial of noun or words which is a specific and deep hierarchical description of source activity. For example, consider that ""cook steak"" is input activity, and sequence is about cooking steak. Ask yourself what steak? Going down with steak can give you more information on animal types. Then you can go deeper if there is information on areas or specific cut of meat(e.g. tomahawk, sirloin, etc). This can result in different types of cuisines finally. The final example can end like ""(cook)(steak)(pork)(loin)(cutlet)"". You cook steak which is pork, which is loin, which is cutlet. Final answer must be given as multiple small brackets enclosing a word. (verb)(noun) is a format used to describe input source activity. (verb)(noun)(noun)....(noun) format is used for final answer called categorized_activity. When you make your answer, start from (verb)(noun) for the input activity, and then order the remaining (noun) from the highest category to the narrowest one!. For our example just return ""(cook)(steak)(pork)(loin)(cutlet)"" and nothing else! Like this!  For this example 'output': ""(cook)(steak)(pork)(loin)(cutlet)"""
    #         }, 
    #         { "role": "user", "content": prompt}
    #             ],
    #     temperature=0.5
    # )
    # categorized_activity = response.choices[0].message.content.strip()
    # return f"Thought: Here is the categorized activity.\n{categorized_activity}"    


    #Ollama Response 
    #     response = ollama.chat(
    #     model = LLM_MODEL_AGENT,
    #     messages=[
    #         {
    #         "role": "system", 
    #         "content": "You predict current user activity based on five input items. Activity MUST be given in one phrase inside a double quote. Answer format is as follows {{action in form of verb}} {{target in form of noun}}"
    #         }
    #     ],
    #     options={
    #         'temperature':0.5
    #     }
    # )
    # activity = response['message']['content']