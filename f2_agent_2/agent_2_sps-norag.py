import sys
import os
import openai
import logging
import json
import ast
import argparse
from dotenv import load_dotenv
import pickle
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
#from langgraph.checkpoint.memory import MemorySaver # Saves everyghing leading to overflow
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import f1_init.agent_init as agent_init
import f1_init.database_init as database_init
import f2_agent.agent_prompt as agent_prompt
import util.util_funcs as util_funcs
import f1_init.constants_init as constants_init
import re

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
        ("system", 
         """You are a helpful taxonomy summarizer that summarizes an action_sequence in input scene_graph, using tools. State your final answer as a string in a section labeled 'Final Answer:'.:
        
            "Final Answer: [Your answer]"
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}, "source_action_sequence": {source_action_sequence}, "source_scene_graph": {source_scene_graph}]

        """),

        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "{query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )

    MESSAGE_ACTIVITY_PREDICTION = [
            {"role": "system", "content": """You are a linguist that summarizes current user source_action_sequence to a single verb and a single noun. Follow these steps.   

             1. Summarize the source_action_sequence to a single verb and a single noun.
             2. Make sure you are only using name of entities, available from source_scene_graph.

            Output Format:
            Return your answer in double quotes, like this:

            Final Answer: "Your Answer"
             
             """}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"}        
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
    source_core_activity = inputs[3]
    query = "make dict of core properties for the given core_activity"
    tool_names =", ".join([t.name for t in tools])    

    AGENT1b_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful classifier that makes property dictionary for each of the noun and verb of the source_core_activity. 
            
         Use the following format for each step:
         
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: [The input to the action]

         When you have enough information, conclude with: Final Answer. Final answer is a dictionary with 5 elements, where a value for a key has only a single word, where every word is enclosed in double quotes:
            
            Final Answer: [Your answer]
        """),
        ("system", "This is the user action sequence: {source_action_sequence}."),
        ("system", "Predicted activity must be able to be performed in this scene: {source_scene_graph}."),
        ("system", "Core activity of the current scene: {source_core_activity}"),
        ("system", "Available tools: {tools}. Actively use retrieval tools to come up with plausible answer."),
        ("system", "Tool names: {tool_names}"),  # for React agents
        ("user", "This is the query for the agent: {query}"),
        ("assistant", "{agent_scratchpad}")  # for React agents
        ])


    MESSAGE_TAXONOMY_CREATION = [
            {"role": "system", "content": 
            """
            You are a taxonomy constructor. Your task is to build a 2-level classification taxonomy based on a given verb and noun (the core activity), using the source_action_sequence and source_scene_graph.

            ### Step-by-step Instructions:

            1. You are given a core activity like:
            "cook steak"

            2. Your goal is to create **a list of two dictionaries**:
            - The first dictionary in the list describes the **verb** ("cook").
            - The second describes the **noun** ("steak").

            3. Each dictionary must contain:
            - Exactly **two key-value pairs**.
            - **Keys** must be **single-word categories** (e.g., "method", "vessel", "ingredient").
            - **Values** must be **single entities** that appear in the source_scene_graph or are inferable from the source_action_sequence.

            4. Choose keys that meaningfully classify the activity or object. 
            - For verbs: Describe how the action is done (e.g., "roasting") and what it uses (e.g., "pan").
            - For nouns: Describe its composition or identity (e.g., "pork") and optional elements (e.g., "garnish").

            5. Values should always be:
            - **Grounded** in the source_scene_graph.
            - **Feasible** and **available** within the scene.

            ### Example Output:

            [
            {
            "manner": "roasting",
            "vessel": "pan"
            },
            {
            "ingredient": "pork",
            "garnish": "celery"
            }
            ]

            ### Final Instructions:

            - ONLY output these two dictionaries — **nothing else**.
            - The order of keys should reflect what’s most important for identifying the action or object.
            - Make sure every key has a valid, single-word value that exists in the source.
            - Avoid repeating keys or using multi-word values.

            Strictly follow this format:

            [
            {
            "key1-for-verb": "value1",
            "key2-for-verb": "value2"
            },
            {
            "key1-for-noun": "value1",
            "key2-for-noun": "value2"
            }
            ]
            """             
             },
            {"role": "user", "content": f"Here is the query:\n{query}\n"},
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the core activity:\n{source_core_activity}\n"}
    ]

    return AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION

def get_agent2a_message(inputs:list):
    """
    func: returns prompt and messages used for agent2a\n
    input: single list: [tools, sequence, scenegraph, target_scene_graph, source_activity_taxonomy]\n
    return: AGENT2a_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]    
    source_activity_taxonomy = inputs[4]
    query = "Construct a common taxonomy that is applicable for both source and target scene graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT2a_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful taxonomy examiner. Your task is to examine whether the input taxonomy for the source_scene_graph is applicable target_scene_graph with common_taxonomy_prediction_tool and return the output taxonomy tool as the common taxonomy. Return the final taxonomy following the format below. AGENT MUST NOT MODIFY TAXONOMY FROM THE TOOLS IN ANY WAY. JUST PASS THE TAXONOMY FROM THE TOOL WITHOUT OMITTING OR CHANGING ANYTHING.   
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}]
        """),

        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )
        # ("system", "This is the source scene graph: {source_scene_graph}."),

    MESSAGE_COMMON_TAXONOMY_PREDICTION = [
        {"role": "system", "content":      
        """
        You are a taxonomy examiner. Your task is to validate a given activity taxonomy against a target_scene_graph and return a filtered version called the common_activity_taxonomy.

        ### Input:
        - **source_activity_taxonomy**: A list of two dictionaries.
        - The first dictionary describes the verb.
        - The second dictionary describes the noun.
        Example:
        [
            {
            "manner": "roasting",
            "vessel": "pan"
            },
            {
            "ingredient": "pork",
            "garnish": "celery"
            }
        ]

        - **source_core_activity**: A string with the verb and noun.
        Example:
        "cook steak"

        ### Task:
        1. For each key-value pair in the source_activity_taxonomy:
        - Check if the value exists in the target_scene_graph or can be generated by any action in the target_scene_graph.
        - If the value **exists**, leave it unchanged.
        - If the value does **not exist**, replace it with **"empty"**.

        2. Do **not** delete or rename any keys.
        - Maintain the exact same structure as the input.
        - Only modify the values.
        - Each dictionary must retain **exactly two key-value pairs**.

        ### Example Output:
        [
        {
            "manner": "roasting",
            "vessel": "empty"
        },
        {
            "ingredient": "empty",
            "garnish": "empty"
        }
        ]

        In this example, "pan", "pork", and "celery" were not found in the target_scene_graph or derivable from it, so they were replaced with "empty".

        ### Final Output Format:
        You must only return a list of **exactly two dictionaries**, each containing **exactly two key-value pairs**. Output format:

        - Return a list of exactly two dictionaries.
        - Each dictionary must contain exactly two key-value pairs.
        - Do not change the order or structure.
        - Do not add explanation or any other text.
        """             
            }, 
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
        {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"}
        ]
 
    return AGENT2a_PROMPT, MESSAGE_COMMON_TAXONOMY_PREDICTION

def get_agent2b_message(inputs:list):
    """
    func: returns prompt and messages used for agent2b\n
    input: single list: [tools, sequence, scenegraph, target_scenegraph, source_taxonomy, common_taxonomy, core_activity]\n
    return: AGENT2b_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]
    source_activity_taxonomy = inputs[4]
    common_activity_taxonomy = inputs[5]
    source_core_activity = inputs[6]
    query = "construct a target_acvity_taxonomy that is applicable to target_scene_graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT2b_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful taxonomy enricher that fills in a missing class of an input taxonomy with appropriate values based on target_scene_graph, using tools.        
         
         Return the final re-filled taxonomy as a target_activity_taxonomy, following the format below. AGENT MUST NOT MODIFY TAXONOMY FROM THE TOOLS IN ANY WAY. JUST PASS THE TAXONOMY FROM THE TOOL WITHOUT OMITTING OR CHANGING ANYTHING.
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}, "target_scene_graph": {target_scene_graph}]
        """),

        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the source scene graph: {source_scene_graph}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "This is the source common activity taxonomy: {common_activity_taxonomy}"),
        ("system", "This is the source core activity: {source_core_activity}"),  
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )
    
    MESSAGE_TARGET_TAXONOMY_EXAMINER = [
            {"role": "system", "content": """You are a taxonomy examiner for a generated target_activity_taxonomy. You're main task is to examine if target_activity_taxonomy follows the context of source_core_acivitiy. Let me give you a step-by-step example of how you function.
             
             First, you receive a target_activity_taxonomy in dictionary form:

            {
             "main ingredient": "pork",
             "preparation method": "roasted", 
             "garnish": "salary",
             "sauce": "mustard",
             "spice":["salted","peppered"]
             }

             You also receive source_core_activity. For this example suppose you have:

             "cook steak"

             For this example, the target_activity_taxnomy can be used for core activity of "cook steak". You will simple return the target_activity_taxonomy.
            
             While for this example, we found candidates for filling in the empty fields, we could encounter target_scene_graph where fulfilling source_core_activity is impossible. In this case, fill EVERY VALUE of the target_activity_taxonomy with "impossible" to make sure target_activity_taxonomy is impossible. 

             DO NOT DELETE OR REFORMAT ANY KEY in the common_activity_taxnomy dictionary in the target_activity_taxonomy"""}, 
   
            {"role": "user", "content": f"Here is the source_core_activity:\n{source_core_activity}\n" },            
        ]    
    
            # {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},            

            # {"role": "user", "content": f"Here is the source scene graph. This is NOT target scene graph:\n{source_scene_graph}\n"},

    MESSAGE_TARGET_TAXONOMY_PREDICTION = [            
        {"role": "system", "content": 
        """
        You are a taxonomy generator for target_scene_graph that checks common_activity_taxonomy and converts it to target_activity_taxonomy. Let me give you a step-by-step example of how you function.

        ### Input:
        - **source_activity_taxonomy**: A list of two dictionaries.
        - The first dictionary describes the verb.
        - The second dictionary describes the noun.
        Example:
        [
            {
            "manner": "roasting",
            "vessel": "pan"
            },
            {
            "ingredient": "pork",
            "garnish": "celery"
            }
        ]

        - **source_core_activity**: A string with the verb and noun.
        Example:
        "cook steak"         

        - **common_activity_taxonomy: A list of two dictionaries.
        - The first dictionary describes the verb.
        - The second dictionary describes the noun.        
        Example:
        [
        {
            "manner": "roasting",
            "vessel": "empty"
        },
        {
            "ingredient": "meat",
            "garnish": "empty"
        }
        ]    

        ### Task:
        1. For each key-value pair in the common_activity_taxonomy:
        - If the value is "empty", check whether a suitable replacement exists in the target_scene_graph that still supports achieving the goal described in source_core_activity.
        - If a suitable replacement exists, replace "empty" with that value.
        - If no suitable value exists, replace "empty" with "False".
        - If the value is not "empty", leave it unchanged.

        2. Do **not** delete or rename any keys.
        - Maintain the exact same structure as the input.
        - Only modify the values.
        - Each dictionary must retain **exactly two key-value pairs**.

        ### Example Output:
        [
        {
            "manner": "roasting",
            "vessel": "pot"
        },
        {
            "ingredient": "chicken",
            "garnish": "False"
        }
        ]                

        Explanation (not to be included in the final output): In this example, "pot" was substituted for "pan" and "chicken" for "meat" based on available entities in the target_scene_graph that can serve the function described in source_core_activity. "garnish" was set to "False" because no suitable alternative was found in the target_scene_graph.

        ### Final Output Format:
        You must only return a list of **exactly two dictionaries**, each containing **exactly two key-value pairs**. Output format:

        - Return a list of exactly two dictionaries.
        - Each dictionary must contain exactly two key-value pairs.
        - Do not change the order or structure.
        - Do not add explanation or any other text.        
        """
        }, 

        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the source_core_activity:\n{source_core_activity}\n" },            
        {"role": "user", "content": f"Here is the common activity taxonomy:\n{common_activity_taxonomy}\n"},     
        {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
        ]

    return AGENT2b_PROMPT, MESSAGE_TARGET_TAXONOMY_PREDICTION, MESSAGE_TARGET_TAXONOMY_EXAMINER

def get_agent3_message(inputs:list):
    """
    get agent3 inputs
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]
    source_activity_taxonomy = inputs[4]
    target_activity_taxonomy = inputs[5]
    source_core_activity = inputs[6]
    query = "predict target_action_sequence that can realize the target_activity_taxonomy in the target_scene_graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT3_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful action planner that constructs an action sequence for the target_scene_graph to achieve the target_activity_taxonomy, using tools. Return the final action_sequence as a target_action_sequence, following the format below. Final answer is a list of strings, each string enclosed in double quotes!:
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}]
        """),
        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the source scene graph: {source_scene_graph}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "This is the target activity taxonomy: {target_activity_taxonomy}"),        
        ("system", "This is the source core activity: {source_core_activity}"),        
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )

    MESSAGE_TARGET_SEQUENCE_PREDICTION = [
        {"role": "system", "content": 
         """
        You are an expert action planner. Your task is to transform a given source_action_sequence so that it fits a target_scene_graph while preserving the original context as closely as possible.

        ### Input:
        - **source_action_sequence**: A list of strings in double quotes.
        - List represents a sequence of actions executed in chronological fashion, as element index increases.
        - Each element is a string instruction.
        Example:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put pan with pork on stove",
            "salt the pork",
            "cut celery",
            "put processed celery on meat",
            "serve the pork"
        ]        

        - **source_core_activity**: A string with the verb and noun.
        Example:
        "cook steak"               

        - **source_activity_taxonomy**: A list of two dictionaries.
        - The first dictionary describes the verb.
        - The second dictionary describes the noun.
        Example:
        [
            {
            "manner": "roasting",
            "vessel": "pan"
            },
            {
            "ingredient": "pork",
            "garnish": "celery"
            }
        ]

        - **target_activity_taxonomy: A list of two dictionaries.
        - The first dictionary describes the verb.
        - The second dictionary describes the noun.        
        Example:
        [
        {
            "manner": "roasting",
            "vessel": "pot"
        },
        {
            "ingredient": "chicken",
            "garnish": "False"
        }
        ]   

        ### Task:
        1. For each element in the source_action_sequence:
        - Check if all objects/entities in the instruction phrase exist in the target_scene_graph.
        - If an entity is missing, replace it with a similar or closest alternative from the target_scene_graph. Consult the source_action_sequence in order to undestand what the missing entity's original role was and look at the source_core_activity to check if the changed action instruction does contribute to making this goal (source_core_activity) possible. The target_activity_taxonomy is the more detailed version of how source_core_activity's noun and verb can be described in the target_scene_graph. Therefore, you are advised to follow the context as described in the target_activity_taxonomy.
        - If no suitable replacement is available, mark the action as "impossible".

        2. Ensure consistency:
        - If you replace an object (e.g., "pan" → "pot"), use that same substitution consistently in all subsequent actions.

        Example transformation:
        Input:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put pan with pork on stove",
            "salt the pork",
            "cut celery",
            "put processed celery on meat",
            "serve the pork"
        ]
        Target Scene Graph does not have pan and pork, but contains: pot, and chicken. Celery or its alternative is nowhere to be found in target_activity_taxonomy and target_scene_graph, so instructions regarding celery is completely wiped out.
        Output:
        [
            "put pot on stove",
            "add oil on pot",
            "turn on the stove",
            "put pot with pork on stove",
            "salt the chicken",
            "impossible",
            "impossible",
            "serve the chicken"
        ]

        3. Delete the action instruction of "impossible". Example:
        [
            "put pot on stove",
            "add oil on pot",
            "turn on the stove",
            "put pot with pork on stove",
            "salt the chicken",
            "serve the chicken"
        ]

        4. Rearrange the sequence if necesary, so that the remaining sequence of instructions achieves source_core_action. Add in new instructions that only uses source_target_scene entities if needed. If all efforts fail to achieve the goal of source_core_action, make your output as a single "False" enclosed in double quotes, inside a list. Example:
        [
            "False"
        ]

        Output Format:
        Return only the final modified list enclosed in double quotes, like this:

        Final Answer: ["action1", "action2", ..., "final action"]

        Do not include any other text or explanation. Follow the format exactly.
        """
        }, 
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
        {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
        {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
        {"role": "user", "content": f"Here is the source activity taxonomy:\n{target_activity_taxonomy}\n"},
        {"role": "user", "content": f"Here is the source core activity:\n{source_core_activity}\n"},            
        ]   
    return AGENT3_PROMPT, MESSAGE_TARGET_SEQUENCE_PREDICTION
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

def activity_prediction(MESSAGE_ACTIVITY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR):
    """Predict an core summary of the user activity based on input"""
    my_temperature = 0.5
    try:
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_ACTIVITY_PREDICTION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_ACTIVITY_PREDICTION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']        

    except Exception as e:
        return f"Error: activity_prediction: {str(e)}"

def predict_activity_taxonomy(MESSAGE_TAXONOMY_CREATION, TOOL_LLM_API, TOOL_LLM_STR):
    """Predict an core summary of the user activity based on input"""
    my_temperature = 0.5
    try:
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_TAXONOMY_CREATION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_TAXONOMY_CREATION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    

    except Exception as e:
        return f"Error: activity_prediction: {str(e)}"

def make_common_taxonomy(MESSAGE_COMMON_TAXONOMY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR):
    """Test if goal of deep activity can be met in current target_scene."""
    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_COMMON_TAXONOMY_PREDICTION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_COMMON_TAXONOMY_PREDICTION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: common_taxonomy_prediction: {str(e)}"

def make_target_activity_taxonomy(MESSAGE_TARGET_TAXONOMY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR):
    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_TARGET_TAXONOMY_PREDICTION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_TARGET_TAXONOMY_PREDICTION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: target_taxonomy_prediction: {str(e)}"
  
def predict_target_action_sequence(MESSAGE_TARGET_SEQUENCE_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR):
    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_TARGET_SEQUENCE_PREDICTION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_TARGET_SEQUENCE_PREDICTION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: action_sequence_prediction: {str(e)}"



# -----------------------
# Tool GET Funcs
# -----------------------
def get_agent1a_tools():
    """
    return tools
    """
    tools=[
    # Tool(
    #     name = "goalstep_retriever_tool",
    #     func = goalstep_information_retriever,
    #     description = "Retrieves relevant activity step information in other scenes."
    # ),
    # Tool(
    #     name = "spatial_retriever_tool",
    #     func = spatial_information_retriver,
    #     description = "Retrieves relevant scene and entity information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
    # ),
    Tool(
        name = "activity_prediction_tool",
        func = lambda _: activity_prediction(MESSAGE_ACTIVITY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR),
        description = "Activity prediction tool, which can summarize the sequential multiple actions into a single verb and a single noun."
    ),
    # Tool(
    #     name = "reorder_activity_taxonomy_tool",
    #     func = lambda _: reorder_activity_taxonomy(MESSAGE_REORDER_TAXONOMY),
    #     description = "reorder_activity_taxonomy, which can reorder the input taxonomy."
    # ),    
    ]
    return tools

def get_agent1b_tools():
    """
    return tools
    """
    tools = [
    # Tool(
    #     name = "goalstep_retriever_tool",
    #     func = goalstep_information_retriever,
    #     description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
    # ),
    # Tool(
    #     name = "spatial_retriever_tool",
    #     func = spatial_information_retriver,
    #     description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
    # ),
    Tool(
        name = "predict_activity_taxonomy_tool",
        func = lambda _: predict_activity_taxonomy(MESSAGE_TAXONOMY_CREATION, TOOL_LLM_API, TOOL_LLM_STR),
        description = "This tool generates a taxonomical description of source action sequence"
    ),
    # Tool(
    #     name = "reorder_activity_taxonomy_tool",
    #     func = lambda _: reorder_activity_taxonomy(MESSAGE_REORDER_TAXONOMY),
    #     description = "This tool reorders activity taxnomy taxonomy"
    # ),
    ]
    return tools

def get_agent2a_tools():
    """
    return tools
    """
    tools = [
        # Tool(
        #     name = "goalstep_retriever_tool",
        #     func = goalstep_information_retriever,
        #     description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
        # ),
        # Tool(
        #     name = "spatial_retriever_tool",
        #     func = spatial_information_retriver,
        #     description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
        # ),
        Tool(
            name = "make_common_taxonomy_tool",
            func = lambda _: make_common_taxonomy(MESSAGE_COMMON_TAXONOMY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR),
            description = "Test if goal of deep activity can be met in current target_scene."
        )
    ]
    return tools

def get_agent2b_tools():
    """
    return tools
    """
    tools = [
        # Tool(
        #     name = "goalstep_retriever_tool",
        #     func = goalstep_information_retriever,
        #     description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
        # ),
        # Tool(
        #     name = "spatial_retriever_tool",
        #     func = spatial_information_retriver,
        #     description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
        # ),
        Tool(
            name = "make_target_activity_taxonomy_tool",
            func = lambda _: make_target_activity_taxonomy(MESSAGE_TARGET_TAXONOMY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR),
            description = "make target activty taxonomy that achieves core activity in the target scene."
        ),    
        ]
    return tools

def get_agent3_tools():
    """
    return tools
    """
    tools = [
        # Tool(
        #     name = "goalstep_retriever_tool",
        #     func = goalstep_information_retriever,
        #     description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
        # ),
        # Tool(
        #     name = "spatial_retriever_tool",
        #     func = spatial_information_retriver,
        #     description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
        # ),
        Tool(
            name = "predict target action sequence",
            func = lambda _: predict_target_action_sequence(MESSAGE_TARGET_SEQUENCE_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR),
            description = "make target action sequence that performs the core activity and target_activity_taxonomy in target_scene."
        ),
        ]    
    return tools

# -----------------------
# Agent Function
# -----------------------
def run_agent_1a(input, agent_llm_chat):
    """"
    func: run agent 1a with source video idx info\n
    input: [tools_1a, AGENT1a_PROMPT, source_action_sequence, source_scene_graph]\n
    output: response
    """
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    TOOLNAMES =", ".join([t.name for t in TOOLS])    

    # AGENT
    QUERY = "Give me an phrase describing the activity of source_action_sequence."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!
    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_llm_chat,
        prompt=AGENT_PROMPT
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

def run_agent_1b(input, agent_llm_chat):
    """"
    func: run agent 1a with source video idx info\n
    input: [tools_1b, AGENT1b_PROMPT, source_action_sequence, source_scene_graph, source_core_activity]\n
    output: response
    """
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    source_core_activity= input[4]
    TOOLNAMES =", ".join([t.name for t in TOOLS])

    QUERY = "Categorically describe source activity in a very specific way."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_llm_chat,
        prompt=AGENT_PROMPT
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
            "source_core_activity": source_core_activity,
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
        },
        config={"max_iterations": 5}
    )
    return response

def run_agent_2a(input, agent_llm_chat):
    """"
    func: run agent 2a\n
    input: [tools_2a, AGENT2a_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy]\n
    output: common_activity_taxonomy
    """    
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    target_scene_graph = input[4]
    source_activity_taxonomy= input[5]

    TOOLNAMES =", ".join([t.name for t in TOOLS])
    QUERY = "Find Common Activity Taxonomy."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_llm_chat,
        prompt=AGENT_PROMPT
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
            "source_action_sequence": source_action_sequence,
            "source_scene_graph": source_scene_graph,   
            "target_scene_graph": target_scene_graph,            
            "source_activity_taxonomy": source_activity_taxonomy,
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

def run_agent_2b(input, agent_llm_chat):
    """"
    func: run agent 2b\n
    input: [tools_2b, AGENT2b_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, common_activity_taxnomy]\n
    output: target_activity_taxonomy
    """    
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    target_scene_graph = input[4]
    source_activity_taxonomy= input[5]
    common_activity_taxonomy= input[6]
    source_core_activity = input[7]

    TOOLNAMES =", ".join([t.name for t in TOOLS])
    QUERY = "Find Target Activity Taxonomy."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_llm_chat,
        prompt=AGENT_PROMPT
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
            "source_action_sequence": source_action_sequence,
            "source_scene_graph": source_scene_graph,   
            "target_scene_graph": target_scene_graph,            
            "source_activity_taxonomy": source_activity_taxonomy,
            "common_activity_taxonomy": common_activity_taxonomy,  
            "source_core_activity": source_core_activity,          
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

def run_agent3(input, agent_llm_chat):
    """"
    func: run agent 3\n
    input: [tools_3, AGENT3_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, target_activity_taxonomy, source_core_taxonomy]\n
    output: target_action_sequence
    """
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    target_scene_graph = input[4]
    source_activity_taxonomy= input[5]
    target_activity_taxonomy= input[6] 
    source_core_activity = input[7]

    TOOLNAMES =", ".join([t.name for t in TOOLS])
    QUERY = "Predict Action Sequence for the target scene graph"    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_llm_chat,
        prompt=AGENT_PROMPT
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
            "source_action_sequence": source_action_sequence,
            "source_scene_graph": source_scene_graph,   
            "target_scene_graph": target_scene_graph,            
            "source_activity_taxonomy": source_activity_taxonomy,
            "target_activity_taxonomy": target_activity_taxonomy, 
            "source_core_activity": source_core_activity,           
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

def run_core_action_test(source_core_activity, target_activity_taxonomy, target_action_sequence, TOOL_LLM_API, TOOL_LLM_STR):    

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": """You are a helpful assistant for testing if source_core_action is summarized representation of target_activity_taxonomy and target_action_sequence. 
            
            source_core_action variable comprises of a single noun and a verb.

            target_activity_taxonomy is a description of the single noun, with 5-level taxonomy of key-value pairs, where key is a classification type, and value is the details on the classification.

            target_action_sequence, is a sequence of actions.

            Your task is twofold
            1. Examine if target_activity_taxonomy is a faithful description of the noun in the source_core_activity.
            2. Examine if target_action_sequence is well represented by the source_core_action, so that source_core_activity can be seen as a goal summarizing the target_action sequence.

            Answer to your question in "yes" and "no" in this format below.:

            taxonomy_representation: [yes or no]
            sequence_representation : [yes or no]
            
            If not, return '''no'''"""},
        {"role": "user", "content":  "Here is the source_core_activity: \n{source_core_activity}\n"},
        {"role": "user", "content":  "Here is the target_activity_taxonomy: \n{target_activity_taxonomy}\n"},
        {"role": "user", "content":  "Here is the target_action_sequence: \n{target_action_sequence}\n"},
    ]
    )

    # Format the prompt
    ("AGENT4: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages(
        source_core_activity=source_core_activity,
        target_activity_taxonomy=target_activity_taxonomy,
        target_action_sequence=target_action_sequence
    )

    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            llm = ChatOpenAI(model=TOOL_LLM_STR, temperature=my_temperature)             
            response = llm.invoke(formatted_messages)
            print(f"my response : {response.content}")   
            return response.content

        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= prompt,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: action_sequence_prediction: {str(e)}"

if __name__ == "__main__":
    # -----------------------
    # TEST SETTINGS
    # -----------------------
    agent_api_name = "openai"
    agent_model_name = "gpt-4.1"
    tool_api_name = "openai"
    tool_model_name = "gpt-4.1"    
    AGENT_LLM_API, AGENT_LLM_STR, AGENT_LLM_CHAT = agent_init.SET_LLMS(agent_api_name, agent_model_name, temperature=0.2)
    TOOL_LLM_API, TOOL_LLM_STR, TOOL_LLM_CHAT = agent_init.SET_LLMS(tool_api_name, tool_model_name, temperature=0.2)

    # PATHS
    BASELINE_FOLDER = "/output-2-sps-norag-0609/"
    PATH_SOURCE_TARGET_OUTPUT = constants_init.PATH_SOURCE_TARGET + BASELINE_FOLDER

    source_folder = constants_init.PATH_AUGMENTATION_v8_source
    target_folder = constants_init.PATH_AUGMENTATION_v8_600
    source_spatial_json_list, target_spatial_json_list = agent_init.get_paired_spatial_json_list_v8(source_folder, target_folder)

    # 
    aug_levels = ['0','0.2','0.4','0.6','0.8','1.0']
    trial_index_levels = ['0th']
    

    # # Make source_idx_list that matches length of the above json list
    source_idx_list = [i for i in range(len(source_spatial_json_list)//(len(aug_levels)*len(trial_index_levels))) for _ in range(len(aug_levels))]
    print(len(source_spatial_json_list))


    # for i in range(0, len(source_list)):
    for i in range(len(source_idx_list)):

        # -----------------------
        # CHECK PATHS
        # -----------------------
        PATH_SOURCEINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_sourceinfo.pkl"
        PATH_TARGETINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_targetinfo.pkl"
        PATH_AGENT1a = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent1a.pkl"
        PATH_AGENT1b = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent1b.pkl"
        PATH_AGENT2a = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent2a.pkl"
        PATH_AGENT2b = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent2b.pkl"
        PATH_AGENT3 = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent3.pkl"
        PATH_AGENT4 = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent4.pkl"  

        # init path bools
        bool_sourceinfo = False
        bool_targetinfo = False
        bool_agent1a = False
        bool_agent1b = False
        bool_agent2a = False
        bool_agent2b = False
        bool_agent3 = False
        # bool_agent4 = False

        # check file with paths
        bool_sourceinfo = agent_init.check_file(PATH_SOURCEINFO)
        bool_targetinfo = agent_init.check_file(PATH_TARGETINFO)
        bool_agent1a = agent_init.check_file(PATH_AGENT1a)
        bool_agent1b = agent_init.check_file(PATH_AGENT1b)
        bool_agent2a = agent_init.check_file(PATH_AGENT2a)
        bool_agent2b = agent_init.check_file(PATH_AGENT2b)
        bool_agent3 = agent_init.check_file(PATH_AGENT3)
        # bool_agent4 = agent_init.check_file(PATH_AGENT4)

        # if every file exist, break from this whole loop
        if bool_sourceinfo and bool_targetinfo and bool_agent1a and bool_agent1b and bool_agent2a and bool_agent2b and bool_agent3:
            continue   
        else:
            print(f"{i} missing")

        # if no file whatsoever, bool_runall is True to run everything without loading
        if not bool_sourceinfo and not bool_targetinfo and not bool_agent1a and not  bool_agent1b and not bool_agent2a and not bool_agent2b and not bool_agent3:
            bool_runall = True
            
        # prepare necessary files
        source_video_idx = source_idx_list[i]
        source_action_sequence, scenegraphnotused = agent_init.get_video_info(source_video_idx)
        source_scene_graph = agent_init.extract_spatial_context(source_spatial_json_list[i])
        target_scene_graph = agent_init.extract_spatial_context(target_spatial_json_list[i])
        source_uid  = source_spatial_json_list[i]['video_id']
        target_uid = target_spatial_json_list[i]['video_id']
        target_equal_ratio  = target_spatial_json_list[i]['target_equal_ratio']

        # sourceinfo and targetinfo
        while not bool_sourceinfo and not bool_targetinfo:
            with open(PATH_SOURCEINFO, 'wb') as f:
                print(f"{i} ")
                dict = {"source_idx": source_video_idx, "source_uid": source_uid, "source_action_sequence": source_action_sequence, "source_scene_graph": source_scene_graph, "target_equal_ratio": target_equal_ratio}
                pickle.dump(dict, f)
                bool_sourceinfo = True

            with open(PATH_TARGETINFO, 'wb') as f:
                print(f"{i} ")
                dict = {"target_idx": (source_video_idx+10)%100, "target_uid": target_uid, "target_scene_graph": target_scene_graph}
                pickle.dump(dict, f)
                bool_targetinfo = True


        # # -----------------------
        # # AGENT1a: PREDICT CORE ACTIVITY
        # # -----------------------
        if bool_agent1a:
            with open(PATH_AGENT1a, 'rb') as f:
                source_core_activity = pickle.load(f)
        else:
            while not bool_agent1a:       
                try:
                    tools_1a = get_agent1a_tools()
                    input_1a_message = [tools_1a, source_action_sequence, source_scene_graph]
                    AGENT1a_PROMPT, MESSAGE_ACTIVITY_PREDICTION = get_agent1a_message(input_1a_message)
                    input_1a_agent = [tools_1a, AGENT1a_PROMPT, source_action_sequence, source_scene_graph]
                    response_1a = run_agent_1a(input_1a_agent, AGENT_LLM_CHAT)
                    source_core_activity = response_1a['output']

                    with open(PATH_AGENT1a, 'wb') as f:
                        pickle.dump(source_core_activity, f)        
                        print(f"agent1a saved: source_core_activity")  
                        bool_agent1a = True 

                except Exception as e:
                    print(f"Agent1a failed at index {i}: {e}")
                    continue


        # -----------------------
        # AGENT1b: PREDICT FULL ACTIVITY TAXONOMY
        # -----------------------
        if bool_agent1b:
            with open(PATH_AGENT1b, 'rb') as f:
                source_activity_taxonomy = pickle.load(f)
        else:
            while not bool_agent1b:        
                try:        
                    tools_1b = get_agent1b_tools()
                    input1b_message = [tools_1b, source_action_sequence, source_scene_graph, source_core_activity]
                    # AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION, MESSAGE_REORDER_TAXONOMY = get_agent1b_message(input1b_message)
                    AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION = get_agent1b_message(input1b_message)
                    input_1b_agent = [tools_1b, AGENT1b_PROMPT, source_action_sequence, source_scene_graph, source_core_activity]
                    response_1b = run_agent_1b(input_1b_agent, AGENT_LLM_CHAT)
                    source_activity_taxonomy = response_1b['output']
                    print(f"1b output {source_activity_taxonomy}")
                    
                    source_activity_taxonomy = re.sub(r"^```json\s*|\s*```$", "", source_activity_taxonomy.strip())
                    source_activity_taxonomy = util_funcs.jsondump_agent_response(source_activity_taxonomy)

                    with open(PATH_AGENT1b, 'wb') as f:
                        pickle.dump(source_activity_taxonomy, f)        
                        print(f"agent1b saved: source_activity_taxonomy")
                        bool_agent1b = True

                except Exception as e:
                    print(f"Agent1b failed at indiex {i}: {e}")
                    continue


        # -----------------------
        # AGENT2a: PREDICT COMMON ACTIVITY TAXONOMY
        # -----------------------            
        # print(source_activity_taxonomy)
        if bool_agent2a:
            with open(PATH_AGENT2a, 'rb') as f:
                common_activity_taxonomy = pickle.load(f)
        else: 
            while not bool_agent2a:       
                try:                      
                    tools_2a = get_agent2a_tools()
                    input2a_message = [tools_2a, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy]
                    AGENT2a_PROMPT, MESSAGE_COMMON_TAXONOMY_PREDICTION=get_agent2a_message(input2a_message)
                    input2a_agent = [tools_2a, AGENT2a_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy]
                    response_2a = run_agent_2a(input2a_agent, AGENT_LLM_CHAT)
                    common_activity_taxonomy = response_2a['output']
                    print(f"2a output {common_activity_taxonomy}")

                    common_activity_taxonomy = re.sub(r"^```json\s*|\s*```$", "", common_activity_taxonomy.strip())
                    common_activity_taxonomy = util_funcs.jsondump_agent_response(common_activity_taxonomy)

                    with open(PATH_AGENT2a, 'wb') as f:
                        pickle.dump(common_activity_taxonomy, f)        
                        print(f"agent2a saved: common_activity_taxonomy")   
                        bool_agent2a = True

                except Exception as e:
                    print(f"Agent2a failed at index {i}: {e}")
                    continue


        # -----------------------
        # AGENT2b: PREDICT TARGET ACTIVITY TAXONOMY
        # -----------------------    
        if bool_agent2b:
            with open(PATH_AGENT2b, 'rb') as f:
                target_activity_taxonomy = pickle.load(f)
        else:       
            while not bool_agent2b:  
                try:                      
                    tools_2b = get_agent2b_tools()
                    input2a_message = [tools_2b, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, common_activity_taxonomy, source_core_activity]    
                    AGENT2b_PROMPT, MESSAGE_TARGET_TAXONOMY_PREDICTION, MESSAGE_TARGET_TAXONOMY_EXAMINER=get_agent2b_message(input2a_message)
                    input2b_agent = [tools_2b, AGENT2b_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, common_activity_taxonomy, source_core_activity]    
                    response_2b = run_agent_2b(input2b_agent, AGENT_LLM_CHAT)
                    target_activity_taxonomy = response_2b['output']
                    print(f"2b output {target_activity_taxonomy}")

                    target_activity_taxonomy = re.sub(r"^```json\s*|\s*```$", "", target_activity_taxonomy.strip())
                    target_activity_taxonomy = util_funcs.jsondump_agent_response(target_activity_taxonomy)
                    
                    with open(PATH_AGENT2b, 'wb') as f:
                        pickle.dump(target_activity_taxonomy, f)        
                        print(f"agent2b saved: target_activity_taxonomy saved")
                        bool_agent2b = True

                except Exception as e:
                    print(f"Agent2b failed at index {i}: {e}")
                    continue
        
        # -----------------------
        # PREDICT TARGET ACTION SEQUENCE
        # -----------------------
        if bool_agent3:
            with open(PATH_AGENT3, 'rb') as f:
                target_action_sequence = pickle.load(f)
        else:            
            while not bool_agent3: 
                try:                      
                    tools_3 = get_agent3_tools()
                    input3_message = [tools_3, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, target_activity_taxonomy, source_core_activity]
                    AGENT3_PROMPT, MESSAGE_TARGET_SEQUENCE_PREDICTION=get_agent3_message(input3_message)       
                    input3_agent = [tools_3, AGENT3_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, target_activity_taxonomy, source_core_activity]   
                    response_3 = run_agent3(input3_agent, AGENT_LLM_CHAT)
                    target_action_sequence = response_3['output']

                    # SERIALIZE TO FORMAT
                    target_action_sequence = re.sub(r"^```json\s*|\s*```$", "", target_action_sequence.strip())
                    target_action_sequence = util_funcs.jsondump_agent_response(target_action_sequence)

                    with open(PATH_AGENT3, 'wb') as f:
                        pickle.dump(target_action_sequence, f)        
                        print(f"agent3 saved: target_action_sequence")
                        bool_agent3 = True

                except Exception as e:
                    print(f"Agent3 failed at index {i}: {e}")
                    continue        



        # # -----------------------
        # # FINAL: TEST FAITHFULNESS TO CORE ACTIVITY
        # # -----------------------     
        # if bool_agent4:
        #     with open(PATH_AGENT4, 'wb') as f:
        #         final_response = pickle.load(f)#    
        # else:
        #     while not bool_agent4:
        #         try:
        #             final_response = run_core_action_test(
        #                 source_core_activity,
        #                 target_activity_taxonomy,
        #                 target_action_sequence,
        #                 agent_api_name,
        #                 agent_model_name
        #             )

        #             print(final_response)
        #             with open(PATH_AGENT4, 'wb') as f:
        #                 pickle.dump(final_response, f)        
        #                 print(f"agent3 saved: output: final answer {final_response}")
        #                 bool_agent4 = True

        #         except Exception as e:
        #             print(f"Agent4 failed at index {i}: {e}")
        #             continue                        