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
from langchain.prompts import ChatPromptTemplate
import f1_init.agent_init as agent_init
import f1_init.database_init as database_init
import f2_agent.agent_prompt as agent_prompt
import util.util_funcs as util_funcs
import f1_init.constants_init as constants_init
import re
import agents_llmfuncs as llmfuncs


#=========================================
# Core Activity Prediction / Sequence Transfer
#=========================================
# def format_scene_graph(scene_graph):
#     '''
#     This function turns raw scene_graph to usable format for LLM
#     '''
#     return "\n".join(
#         f"Object ID: {obj['object_id']}, Name: {obj['object_name']}, "
#         f"Status: {obj['init_status'].get('status')}, "
#         f"Container: {obj['init_status'].get('container')}"
#         for obj in scene_graph
#     )

def format_scene_graph(scene_graph):
    '''
    This function turns raw scene_graph to usable format for LLM
    '''
    def extract_status_info(init_status):
        if isinstance(init_status, dict):
            status = init_status.get('status', 'unknown')
            container = init_status.get('container', 'none')
        else:
            status = init_status
            container = 'none'
        return status, container

    return "\n".join(
        f"Object ID: {obj['object_id']}, Name: {obj['object_name']}, "
        f"Status: {extract_status_info(obj['init_status'])[0]}, "
        f"Container: {extract_status_info(obj['init_status'])[1]}"
        for obj in scene_graph
    )

def format_goalstep_examples(examples):
    '''
    Format goalstep example Documents into readable string input for LLM
    '''
    formatted = []
    for doc in examples:
        meta = doc.metadata
        formatted.append(
            f"Video UID: {meta.get('video_uid')}\n"
            f"Step Category: {meta.get('step_category')}\n"
            f"Step Description: {meta.get('step_description')}\n"
            f"Segment ID: {meta.get('segment_id')}, Time: {meta.get('start_time')} - {meta.get('end_time')}\n"
            f"Page Content: {doc.page_content}\n"
            "---"
        )
    return "\n".join(formatted)

# def format_spatial_examples(examples):
#     '''
#     Format spatial example Documents (initial state) into readable string input for LLM
#     '''
#     formatted = []
#     for doc in examples:
#         meta = doc.metadata
#         formatted.append(
#             f"Video UID: {meta.get('video_uid')}\n"
#             f"Initial State Object Graph:\n{doc.page_content.strip()}\n"
#             "---"
#         )
#     return "\n".join(formatted)

def format_spatial_examples(docs):
    formatted = []
    for doc in docs:
        video_uid = doc.metadata.get("video_uid", "unknown")
        content = doc.page_content
        # Escape curly braces
        content = content.replace("{", "{{").replace("}", "}}")
        formatted_doc = f"Video UID: {video_uid}\nInitial State Object Graph:\n{content}"
        formatted.append(formatted_doc)
    return "\n---\n".join(formatted)

def run_agent1a_llm_norag(source_action_sequence, source_scene_graph, TOOL_LLM_API, TOOL_LLM_STR, temperature):

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", 
         "content": 
        """
        You are a linguist that summarizes current user source_action_sequence as a list which contains a single verb and a single noun as separate elements. Follow these steps.   

        1. Summarize the source_action_sequence to a single verb and a single noun.
        2. Make sure you are only using name of entities, available from source_scene_graph.
        3. When summarizing to a noun and a verb, you must only use name of entities, available from source_scene_graph, or verb only possible with interacting with entities inside source_scene_graph.        

        Output Format:
        You Must return your answer as a list with the single verb and a single noun as elements. Verb and Noun should be enclosed in double quotes in the list.Follow the format below:

        ["verb", "noun"]
        """
        },
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"}    
    ]
    )

    # Format the prompt
    #("AGENT1a: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages()

    # Invoke (OpenAI / Ollama)
    my_temperature = temperature
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

# EXAMPLES NEEED FORMATTING TOO!
def run_agent1a_llm_rag(source_action_sequence, source_scene_graph, goalstep_example, spatial_example, TOOL_LLM_API, TOOL_LLM_STR, temperature):

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", 
         "content": 
        """
        You are a linguist that summarizes current user source_action_sequence as a list which contains a single verb and a single noun as separate elements. Follow these steps.   

        1. Summarize the source_action_sequence to a single verb and a single noun.
        2. Make sure you are only using name of entities, available from source_scene_graph or verb only possible with entities inside source_scene_graph.
        3. You can use relevant information from goalstep_example for how sequences are summarized into description or scenarios.
        4. You can see how certain scenes can support certain sequence of actions in spatial_example, but do not use entities.
        5. When summarizing to a noun and a verb, you must only use name of entities, available from source_scene_graph, or verb only possible with interacting with entities inside source_scene_graph.

        Output Format:
        You Must return your answer as a list with the single verb and a single noun as elements. Verb and Noun should be enclosed in double quotes in the list. Follow the format below:

        ["verb", "noun"]
        """
        },
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"},
        {"role": "user", "content": f"Here is thhe action sequence examples from similar environment:\n {goalstep_example}\n"},
        {"role": "user", "content": f"Here is the scene graph from similar environment:\n {spatial_example}\n"}             
    ]
    )

    # Format the prompt
    #("AGENT1a: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages()

    # Invoke (OpenAI / Ollama)
    my_temperature = temperature
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


#=========================================
# Sequence Transfer (s2s, sgs)
#=========================================
def run_agent3_llm_s2s(source_action_sequence, source_scene_graph, target_scene_graph, TOOL_LLM_API, TOOL_LLM_STR, temperature):

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": 
        """
        You are an expert action planner. Your task is to transform a given source_action_sequence to a target_action_sequence, so that it fits a target_scene_graph while preserving the original context as closely as possible.

        Follow this 3-step process:

        1. You will receive a list of actions (source_action_sequence), each as a string instruction. Example:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put meat on stove",
            "salt the meat"
        ]

        2. For each action instruction:
        - Check if all objects/entities in the instruction phrase exist in the target_scene_graph.
        - If an entity is missing, replace it with a similar or closest alternative from the target_scene_graph. Also change the verb accordingly for the changed entity. Consult the source_action_sequence in order to undestand what the missing entity's original role was.
        - If no suitable replacement is available, mark the action as "impossible".

        3. Ensure consistency:
        - If you replace an object (e.g., "pan" → "pot"), use that same substitution consistently in all subsequent actions.

        Example transformation:
        Input:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put meat on stove",
            "salt the meat"
        ]
        Target Scene Graph does not have pan and meat, but contains: pot, stove, oil
        Output:
        [
            "put pot on stove",
            "add oil on pot",
            "turn on the stove",
            "impossible",
            "impossible"
        ]

        4. Delete the action instruction of "impossible". Example:
        [
            "put pot on stove",
            "add oil on pot",
            "turn on the stove",
        ]

        5. Rearrange the sequence if necesary, so that the remaining sequence of instructions achieves similar context as source_action_sequence. Add in new instructions that only uses source_target_scene entities if needed. If all efforts fail to achieve the goal of source_action_sequence, make your output as a single "False" enclosed in double quotes, inside a list. Example:
        [
            "False"
        ]

        Output Format:
        The final answer is a single list of instructions, each enclosed in double quotes, and must stick to the format below:

        ["action1", "action2", ..., "final action"]
        """
        }, 
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
        {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
        ])

    # Format the prompt
    #("AGENT3: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages()

    # Invoke (OpenAI / Ollama)
    my_temperature = temperature
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



def run_agent3_llm_norag(source_action_sequence, source_scene_graph, source_core_activity, target_scene_graph, TOOL_LLM_API, TOOL_LLM_STR, temperature):

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": 
        """
        You are an expert action planner. Your task is to transform a given source_action_sequence to a target_action_sequence, so that it fits a target_scene_graph while preserving the original context as closely as possible.

        Follow this 3-step process:

        1. You will receive a list of actions (source_action_sequence), each as a string instruction. Example:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put meat on stove",
            "salt the meat"
        ]

        2. You will also receive a goal for the actions (source_core_activity), which comprises of a verb and a noun. Example:

        "cook steak"

        3. For each action instruction:
        - Check if all objects/entities in the instruction phrase exist in the target_scene_graph.
        - If an entity is missing, replace it with a similar or closest alternative from the target_scene_graph. Also change the verb accordingly for the changed entity. Consult the source_action_sequence in order to undestand what the missing entity's original role was and look at the source_core_activity to check if the changed action instruction still preseerves the goal (source_core_activity).
        - If no suitable replacement is available, mark the action as "impossible".

        4. Ensure consistency:
        - If you replace an object (e.g., "pan" → "pot"), use that same substitution consistently in all subsequent actions.

        Example transformation:
        Input:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put meat on stove",
            "salt the meat"
        ]
        Target Scene Graph does not have pan and meat, but contains: pot, stove, oil
        Output:
        [
            "put pot on stove",
            "add oil on pot",
            "turn on the stove",
            "impossible",
            "impossible"
        ]

        5. Delete the action instruction of "impossible". Example:
        [
            "put pot on stove",
            "add oil on pot",
            "turn on the stove",
        ]

        6. Rearrange the sequence if necesary, so that the remaining sequence of instructions achieves source_core_activity. Add in new instructions that only uses source_target_scene entities if needed. If all efforts fail to achieve the goal of source_core_activity, make your output as a single "False" enclosed in double quotes, inside a list. Example:
        [
            "False"
        ]

        Output Format:
        The final answer is a single list of instructions, each enclosed in double quotes, and must stick to the format below:

        ["action1", "action2", ..., "final action"]
        """
        }, 
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the source core activity:\n{source_core_activity}\n"},
        {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
        {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
        ])

    # Format the prompt
    #("AGENT3: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages()

    # Invoke (OpenAI / Ollama)
    my_temperature = temperature
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


def run_agent3_llm_sps_norag(source_action_sequence, source_scene_graph, source_core_activity, target_scene_graph, source_activity_taxonomy, target_activity_taxonomy, TOOL_LLM_API, TOOL_LLM_STR, temperature):

    prompt = [
        {"role": "system", "content": 
        """
        You are an expert action planner. Your task is to transform a given source_action_sequence to a target_action_sequence, so that it fits a target_scene_graph while preserving the original context as closely as possible.

        ### Input:
        - **source_activity_taxonomy**: A list of two dictionaries.
        - The first dictionary describes the verb.
        - The second dictionary describes the noun.
        Example:
        [
            {
            "cooking method": "roasting",
            "cooking vessel": "pan"
            },
            {
            "steak main ingredient": "pork",
            "steak garnish": "celery"
            }
        ]


        - **target_activity_taxonomy**: A list of two dictionaries, that is holds same keys as source_activity_taxonomy, and tries to fill equal or most similar values with objects or states available from target_scene_graph.
        - The first dictionary describes the verb.
        - The second dictionary describes the noun.
        Example:
        [
            {
                "cooking method": "roasting",
                "cooking vessel": "pot"
            },
            {
                "steak main ingredient": "chicken",
                "steak garnish": "empty"
            }
        ]   

        - **source_core_activity**: A list of two string elements, each for the verb and the noun description in the source_activity_taxonomy.
        Example:
        ["cook" ,"steak"]        


        -**source_action_sequence**: A list of action instructions for the source_scene_graph, executed sequentially to achieve the goal of source_core_activity.
        Example:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put meat on stove",
            "salt the meat"
        ]

        Follow this 3-step process:

        1. You will receive a list of actions (source_action_sequence), each as a string instruction. Example:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put meat on stove",
            "salt the meat"
        ]

        2. You will also receive a goal for the actions (source_core_activity), which comprises of a verb and a noun. Example:

        ["cook", "steak"]

        3. For each action instruction:
        - Check if all objects/entities in the instruction phrase exist in the target_scene_graph.
        - If an entity is missing, replace it with a similar or closest alternative from the target_scene_graph. Also change the verb accordingly for the changed entity. 
        - Consult the source_action_sequence, source_core_activity, and source_activity_taxonomy in order to undestand what the missing entity's original role was.
        - The modified action must comply with the context in target_activity_taxonomy.
        - If no suitable replacement is available, mark the action as "impossible".

        4. Ensure consistency:
        - If you replace an object (e.g., "pan" → "pot"), use that same substitution consistently in all subsequent actions.

        Example transformation:
        Input:
        [
            "put pan on stove",
            "add oil on pan",
            "turn on the stove",
            "put meat on stove",
            "salt the meat"
        ]
        Target Scene Graph does not have pan and meat, but contains: pot, stove, oil
        Output:
        [
            "put pot on stove",
            "add oil on pot",
            "turn on the stove",
            "impossible",
            "impossible"
        ]

        5. Delete the action instruction of "impossible". Example:
        [
            "put pot on stove",
            "add oil on pot",
            "turn on the stove",
        ]

        6. Rearrange the sequence if necesary, so that the remaining sequence of instructions achieves source_core_activity. Add in new instructions that only uses source_target_scene entities if needed. If all efforts fail to achieve the goal of source_core_activity, make your output as a single "False" enclosed in double quotes, inside a list. Example:
        [
            "False"
        ]

        Output Format:
        The final list of instructions, each enclosed in double quotes, must stick to the format below:

        ["action1", "action2", ..., "final action"]
        """
        }, 
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the source core activity:\n{source_core_activity}\n"},
        {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
        {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
        ]   

    # Format the prompt
    #("AGENT3: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages(
        source_action_sequence=source_action_sequence,
        source_scene_graph=source_scene_graph,
        source_core_activity = source_core_activity,
        target_scene_graph = target_scene_graph
    )

    # Invoke (OpenAI / Ollama)
    my_temperature = temperature
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


#=========================================
# Property Prediction
#=========================================
def run_agent1b_llm_norag(source_action_sequence, source_scene_graph, source_core_activity, target_scene_graph, TOOL_LLM_API, TOOL_LLM_STR, temperature):

    prompt = [
        {"role": "system", "content": 
        """
        You are a taxonomy constructor. Your task is to build two 2-level classification taxonomies for a given verb and a noun (the core activity), using the source_action_sequence and source_scene_graph.

        ### Step-by-step Instructions:

        1. You are given a source_core_activity like:
        ["cook", "steak"]

        2. Your goal is to create **a list of two dictionaries**:
        - The first dictionary in the list describes the **verb** ("cook").
        - The second describes the **noun** ("steak").

        3. Each dictionary must contain:
        - Exactly **two key-value pairs**.
        - **Keys** must take form of the core_activity noun or verb acting as adjective to give context for a key (e.g., "cooking method", "cooking vessel", "steak main ingredient").
        - **Values** must be **single entities** that appear in the source_scene_graph or are inferable from the source_action_sequence.

        4. Choose keys that meaningfully classify the source_core_activity words, referring to the detailed action sequence in source_action_sequence, and objects in source_scene_graph. For example of ["cook", "steak"]:
        - For verb: Describe how the action of "cook" is done (e.g., "roasting") and what it uses (e.g., "pan").
        - For noun: Describe composition or identity of "steak" (e.g., "pork") and optional elements (e.g., "garnish").

        5. Values should always be:
        - **Grounded** in the source_scene_graph.
        - **Feasible** and **available** within the scene.

        ### Example Output:

        [
        {
        "cooking method": "roasting",
        "cooking vessel": "pan"
        },
        {
        "steak main ingredient": "pork",
        "steak garnish": "celery"
        }
        ]

        ### Final Instructions:

        - ONLY output these two dictionaries in a list— **nothing else**.
        - The order of keys in each dictionary should reflect what’s most important for identifying the action or object.
        - Make sure every key has a valid, single-word value that exists in the source_scene_graph or activity possible within the source_scene_graph.
        - Avoid repeating keys or using multi-word values.

        Strictly follow this format for final answer:

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
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"},
        {"role": "user", "content": f"Here is the core activity:\n{source_core_activity}\n"}
        ]   

    # Format the prompt
    #("AGENT1b: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages(
        source_action_sequence=source_action_sequence,
        source_scene_graph=source_scene_graph,
        source_core_activity = source_core_activity,
    )

    # Invoke (OpenAI / Ollama)
    my_temperature = temperature
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


#=========================================
# Property Transfer
#=========================================
def run_agent2a_llm_norag(source_action_sequence, source_scene_graph, source_core_activity, source_activity_taxonomy, target_scene_graph, TOOL_LLM_API, TOOL_LLM_STR, temperature):

    prompt = [
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
            "cooking method": "roasting",
            "cooking vessel": "pan"
            },
            {
            "steak main ingredient": "pork",
            "steak garnish": "celery"
            }
        ]

        - **source_core_activity**: A list of two string elements, each for the verb and the noun description in the source_activity_taxonomy.
        Example:
        ["cook" ,"steak"]

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
            "cooking method": "roasting",
            "cooking vessel": "empty"
        },
        {
            "steak main ingredient": "empty",
            "steak garnish": "empty"
        }
        ]

        In this example, "pan", "pork", and "celery" were not found in the target_scene_graph or derivable from it, so they were replaced with "empty".

        ### Final Output Format:
        You must only return a list of **exactly two dictionaries**, each containing **exactly two key-value pairs**. Output format:

        - Return a list of exactly two dictionaries.
        - Each dictionary must contain exactly two key-value pairs.
        - Do not change the order or structure.
        - Do not add explanation or any other text.

        Strictly follow this format for final answer:

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
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
        {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"}
        ]   

    # Format the prompt
    #("AGENT2a FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages(
        source_activity_taxonomy=source_activity_taxonomy,
    )

    # Invoke (OpenAI / Ollama)
    my_temperature = temperature
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

def run_agent2b_llm_norag(source_action_sequence, source_scene_graph, source_core_activity, target_scene_graph, source_activity_taxonomy, common_activity_taxonomy, TOOL_LLM_API, TOOL_LLM_STR, temperature):

    prompt = [
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
            "cooking method": "roasting",
            "cooking vessel": "pan"
            },
            {
            "steak main ingredient": "pork",
            "steak garnish": "celery"
            }
        ]


        - **common_activity_taxonomy**: A list of two dictionaries, that is holds same keys as source_activity_taxonomy, and tries to fill values with objects or states available from target_scene_graph.
        - The first dictionary describes the verb.
        - The second dictionary describes the noun.
        Example:
        [
            {
                "cooking method": "roasting",
                "cooking vessel": "empty"
            },
            {
                "steak main ingredient": "meat",
                "steak garnish": "empty"
            }
        ]   


        - **source_core_activity**: A list of two string elements, each for the verb and the noun description in the source_activity_taxonomy.
        Example:
        ["cook" ,"steak"]

        ### Task:
        1. For each key-value pair in the common_activity_taxonomy:
        - If the value is "empty", check whether a suitable replacement exists in the target_scene_graph that still supports achieving the goal described in source_core_activity, and preserves context of corresponding value in the source_activity_taxonomy.
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
                "cooking method": "roasting",
                "cooking vessel": "pot"
            },
            {
                "steak main ingredient": "chicken",
                "steak garnish": "False"
            }
        ]   

        In this example, "pan", "pork", and "celery" were not found in the target_scene_graph or derivable from it, so they were replaced with "empty".

        ### Final Output Format:
        You must only return a list of **exactly two dictionaries**, each containing **exactly two key-value pairs**. Output format:

        - Return a list of exactly two dictionaries.
        - Each dictionary must contain exactly two key-value pairs.
        - Do not change the order or structure.
        - Do not add explanation or any other text.

        Strictly follow this format for final answer:

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
        {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
        {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
        {"role": "user", "content": f"Here is the source core activity:\n{source_core_activity}\n"},
        {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
        {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
        {"role": "user", "content": f"Here is the common activity taxonomy:\n{common_activity_taxonomy}\n"},                
        ]   

    # Format the prompt
    ("AGENT3: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages(
    )

    # Invoke (OpenAI / Ollama)
    my_temperature = temperature
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

