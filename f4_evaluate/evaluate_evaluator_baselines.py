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

import numpy as np
from numpy.linalg import norm
from fastdtw import fastdtw
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from scipy.spatial.distance import cosine

def get_evaluations_taxseq(BASELINE_FOLDER):
    PATH_SOURCE_TARGET_OUTPUT = constants_init.PATH_SOURCE_TARGET + BASELINE_FOLDER
    source_spatial_json_list, target_spatial_json_list, aug_levels = agent_init.get_paired_spatial_json_list(constants_init.PATH_AUGMENTATION_v6)
    source_idx_list = [i for i in range(len(source_spatial_json_list)//len(aug_levels)) for _ in range(len(aug_levels))]

    result_list = []
    for i in range(len(source_idx_list)):
        result_dict = {}
        # -----------------------
        # CHECK PATHS
        # -----------------------
        PATH_SOURCEINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_sourceinfo.pkl" #source idx, seq, uid
        PATH_TARGETINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_targetinfo.pkl" #target uid
        PATH_AGENT1a = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent1a.pkl" #core activity
        PATH_AGENT1b = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent1b.pkl" #source tax
        PATH_AGENT2b = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent2b.pkl"#target tax
        PATH_AGENT3 = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent3.pkl" #target seq
        # -----------------------
        # GET FILES
        # -----------------------
        source_video_idx = source_idx_list[i]
                
        source_uid  = source_spatial_json_list[i]['video_id']
        target_uid = target_spatial_json_list[i]['video_id']   
        spatial_similarity  = target_spatial_json_list[i]['spatial_similarity']     

        source_sequence = agent_init.get_video_info(source_video_idx)
        source_taxonomy = load_file(PATH_AGENT1b)
        core_activity = load_file(PATH_AGENT1a)

        target_sequence = load_file(PATH_AGENT3)
        target_taxonomy = load_file(PATH_AGENT2b)
        target_scene_graph = agent_init.extract_spatial_context(target_spatial_json_list[i])

        current_dict = {'idx':i, "source_uid":source_uid, "target_uid":target_uid, "inclusion_ratio":spatial_similarity, "source_sequence":source_sequence, "source_taxonomy":source_taxonomy, "core_activity":core_activity, "target_sequence":target_sequence,"target_taxonomy":target_taxonomy, "target_scene_graph":target_scene_graph}

        #TODO: entity check with llm
        entity_check_tax = entity_check_tax(current_dict)
        entity_check_seq = entity_check_seq(current_dict)

        #bert, openai_embedding-small, embedding-large
        similarity_tax1 = compute_weighted_tax_similarity(current_dict, "sbert")
        similarity_tax2 = compute_weighted_tax_similarity(current_dict, "text-embedding-3-small")
        similarity_tax3 = compute_weighted_tax_similarity(current_dict, "text-embedding-3-large")
        similarity_seq1 = compute_dtw_seq_similarity(current_dict, "sbert")
        similarity_seq2 = compute_dtw_seq_similarity(current_dict, "text-embedding-3-small")
        similarity_seq3 = compute_dtw_seq_similarity(current_dict, "text-embedding-3-large")

        #TODO: core check for tax sequence
        core_check_tax = core_check(current_dict, "openai", "gpt-4.1", "tax")
        core_check_seq = core_check(current_dict, "openai", "gpt-4.1", "seq")

        # metadata
        result_dict["idx"]=i
        result_dict["baseline"]=BASELINE_FOLDER
        result_dict["source_uid"]=source_uid
        result_dict["target_uid"]=target_uid

        # stuff used for human intervention
        result_dict["target_taxonomy"]=target_taxonomy
        result_dict["target_sequence"]=target_sequence
        result_dict["core_activity"]=core_activity
        
        # previous evaluator block
        result_dict["entity_check_tax"]=entity_check_tax
        result_dict["entity_check_seq"]=entity_check_seq
        result_dict["sim_tax1"]=similarity_tax1
        result_dict["sim_tax2"]=similarity_tax2
        result_dict["sim_tax3"]=similarity_tax3
        result_dict["sim_seq1"]=similarity_seq1
        result_dict["sim_seq2"]=similarity_seq2
        result_dict["sim_seq3"]=similarity_seq3
        result_dict["core_check_tax"]=core_check_tax
        result_dict["core_check_seq"]=core_check_seq
        result_dict["human_check_tax"]=False
        result_dict["human_check_seq"]=False

        result_list.append(result_dict)

    return result_list

def get_evaluations_seq(BASELINE_FOLDER):
    PATH_SOURCE_TARGET_OUTPUT = constants_init.PATH_SOURCE_TARGET + BASELINE_FOLDER
    source_spatial_json_list, target_spatial_json_list, aug_levels = agent_init.get_paired_spatial_json_list(constants_init.PATH_AUGMENTATION_v6)
    source_idx_list = [i for i in range(len(source_spatial_json_list)//len(aug_levels)) for _ in range(len(aug_levels))]

    result_list = []
    for i in range(len(source_idx_list)):
        result_dict = {}
        # -----------------------
        # CHECK PATHS
        # -----------------------
        PATH_SOURCEINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_sourceinfo.pkl" #source idx, seq, uid
        PATH_TARGETINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_targetinfo.pkl" #target uid
        PATH_AGENT1a = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent1a.pkl" #core activity
        PATH_AGENT3 = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent3.pkl" #target seq
        # -----------------------
        # GET FILES
        # -----------------------
        source_video_idx = source_idx_list[i]
                
        source_uid  = source_spatial_json_list[i]['video_id']
        target_uid = target_spatial_json_list[i]['video_id']   
        spatial_similarity  = target_spatial_json_list[i]['spatial_similarity']     

        source_sequence = agent_init.get_video_info(source_video_idx)
        core_activity = load_file(PATH_AGENT1a)

        target_sequence = load_file(PATH_AGENT3)
        target_scene_graph = agent_init.extract_spatial_context(target_spatial_json_list[i])

        current_dict = {'idx':i, "source_uid":source_uid, "target_uid":target_uid, "inclusion_ratio":spatial_similarity, "source_sequence":source_sequence, "core_activity":core_activity, "target_sequence":target_sequence,"target_scene_graph":target_scene_graph}

        #TODO: entity check with llm
        entity_check_seq = entity_check_seq(current_dict)

        #sbert, openai_embedding-small, embedding-large
        similarity_seq1 = compute_dtw_seq_similarity(current_dict, "sbert")
        similarity_seq2 = compute_dtw_seq_similarity(current_dict, "text-embedding-3-small")
        similarity_seq3 = compute_dtw_seq_similarity(current_dict, "text-embedding-3-large")

        #TODO: core check for sequence
        core_check_seq = core_check(current_dict, "openai", "gpt-4.1", "seq")

        # metadata
        result_dict["idx"]=i
        result_dict["baseline"]=BASELINE_FOLDER
        result_dict["source_uid"]=source_uid
        result_dict["target_uid"]=target_uid

        # stuff used for human intervention
        result_dict["target_sequence"]=target_sequence
        result_dict["core_activity"]=core_activity
        
        # previous evaluator block
        result_dict["entity_check_seq"]=entity_check_seq
        result_dict["sim_seq1"]=similarity_seq1
        result_dict["sim_seq2"]=similarity_seq2
        result_dict["sim_seq3"]=similarity_seq3
        result_dict["core_check_seq"]=core_check_seq
        result_dict["human_check_tax"]=False
        result_dict["human_check_seq"]=False

        result_list.append(result_dict)

    return result_list

#-----------------------------------------------------------
# Target Scene Entity Check LLM / Core activity check
#-----------------------------------------------------------
def load_file(path):
    '''
    load file, if no file return none
    '''
    try:
        with open(path, "rb") as f:
            return pickle.load(f)  # try loading to ensure file is not empty or corrupted
    except (EOFError, FileNotFoundError, PermissionError, IsADirectoryError, pickle.UnpicklingError):
        return None 

def entity_check(current_dict, TOOL_LLM_API, TOOL_LLM_STR, mode:str):
    '''
    func return bool on entity_check\n
    input: current_dict, mode: "tax" or "seq"\n
    '''
    source_core_activity = current_dict['core_activity']
    target_activity_taxonomy = current_dict['target_taxonomy']
    target_action_sequence = current_dict['target_sequence']
    target_scene_graph = current_dict['target_scene_graph']

    # entity check for taxonomy
    prompt1 = ChatPromptTemplate.from_messages([
        {"role": "system", "content": """You are a helpful assistant that checks if every of the target_activity_taxonomy values can be found in scene graph and returns yes or no.

    
         -source_core_activity is a noun-verb pair that acts as a final goal.
            
         -target_taxonomy is a dictionary with five key-value pairs. This describes the noun in the source_core_acvitiy.
            
         -target_scene_graph is a list of dictionaries, each dictionary holding information about an object.
            
         -object_id and object_name can be found in the dictionary from 'object_id' and 'object_name' key, respectively.

            1. For each key-value pair of target_taxonomy, see if value of the target_taxonomy can be found in the target_scene_graph, or be realized by actions using objects inside the scene graph.
         
            2. To do this, search dictionaries of target_scene_graph to find equivalent object_name or object_names that are crucial in realizing the value in the target taxonomy. 
         
            3. object_name in target_scene_graph doesn't have to be completely same string-by-string with value from taxonomy, but should be equivalent in meaning.
         
         -If every values for the keys in the taxonomy can be realized by the target_scene_graph objects return '''True'''
         
         -If not, return '''False'''"""},
        {"role": "user", "content":  "Here is the source_core_activity: \n{source_core_activity}\n"},            
        {"role": "user", "content":  "Here is the target_activity_taxonomy: \n{target_activity_taxonomy}\n"},
        {"role": "user", "content":  "Here is the target_scene_graph: \n{target_scene_graph}\n"},
    ]
    )

    # entity check for sequence
    prompt2 = ChatPromptTemplate.from_messages([
        {"role": "system", "content": """You are a helpful assistant that checks if every of the target_activity_taxonomy values can be found in scene graph and returns yes or no.

    
         -source_core_activity is a noun-verb pair that acts as a final goal of target_action_sequence.
            
         -target_action_sequence is a list of string items, where each item describes a step. The strings are executed in sequential order to achieve the goal of source_core_activity.
            
         -target_scene_graph is a list of dictionaries, each dictionary holding information about an object. 'object_id' key holds object_id, while 'object_name' holds name of the object inside the target_scene.

         -For each step in the target_action_sequence, examine if this step can be executed only using items in the target_scene_graph
        
         -If every steps inside the target_action_sequence can be executed with only objects in the target_scene_graph return '''True'''
         
         -If not, return '''False'''"""},
        {"role": "user", "content":  "Here is the source_core_activity: \n{source_core_activity}\n"},            
        {"role": "user", "content":  "Here is the target_action_sequence: \n{target_action_sequence}\n"},
        {"role": "user", "content":  "Here is the target_scene_graph: \n{target_scene_graph}\n"},
    ]
    )
    
    if mode == "tax":
        prompt = prompt1
    elif mode == "seq":
        prompt = prompt2
    else:
        print(f"entity check: mode should be either tax or seq")

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

def core_check(current_dict, TOOL_LLM_API, TOOL_LLM_STR, mode:str):
    '''
    func return bool on entity_check\n
    input: current_dict, mode: "tax" or "seq"\n
    '''
    source_core_activity = current_dict['core_activity']
    target_activity_taxonomy = current_dict['target_taxonomy']
    target_action_sequence = current_dict['target_sequence']

    # entity check for taxonomy
    prompt1 = ChatPromptTemplate.from_messages([
        {"role": "system", "content": """You are a helpful assistant that checks if every of the target_activity_taxonomy values can be found in scene graph and returns yes or no.
    
         -source_core_activity is a noun-verb pair that acts as a final goal.
            
         -target_taxonomy is a dictionary with five key-value pairs. This describes the noun in the source_core_acvitiy.
         
         -If target_taxonomy is a description of noun of source_core_activity return '''True''' as your final answer.
         
         -If not, return '''False'''"""},
        {"role": "user", "content":  "Here is the source_core_activity: \n{source_core_activity}\n"},            
        {"role": "user", "content":  "Here is the target_activity_taxonomy: \n{target_activity_taxonomy}\n"},
    ]
    )

    # entity check for sequence
    prompt2 = ChatPromptTemplate.from_messages([
        {"role": "system", "content": """You are a helpful assistant that checks if every of the target_activity_taxonomy values can be found in scene graph and returns yes or no.

         -source_core_activity is a noun-verb pair that acts as a final goal of target_action_sequence.
            
         -target_action_sequence is a list of string items, where each item describes a step. The strings are executed in sequential order to achieve the goal of source_core_activity.    
        
         -If source_core_activity is a faithful summarized representation of target_action_sequence return '''True''' as your final answer.
         
         -If not, return '''False'''"""},
        {"role": "user", "content":  "Here is the source_core_activity: \n{source_core_activity}\n"},            
        {"role": "user", "content":  "Here is the target_action_sequence: \n{target_action_sequence}\n"},
    ]
    )
    
    if mode == "tax":
        prompt = prompt1
    elif mode == "seq":
        prompt = prompt2
    else:
        print(f"entity check: mode should be either tax or seq")

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

#-----------------------------------------------------------
# Similarity for Tax(Weight) and Sequence(DTW)
#-----------------------------------------------------------
def normalize_to_string(value):
    '''
    used so to cope with value from a dictionary that happens to be inside a list bracket    
    '''
    # If it's a list and the first element is a string, return that
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
        return value[0]
    # If it's already a string, return it
    elif isinstance(value, str):
        return value
    # Fallback for anything else
    else:
        return str(value)
    
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return sklearn_cosine(vec1, vec2)[0][0]

def safe_parse_sequence(seq):
    if seq is None:
        return []
    if isinstance(seq, list):
        return seq
    try:
        return ast.literal_eval(seq)
    except (ValueError, SyntaxError):
        print("taxonomy safe parse value error")
        return []

def safe_parse_taxonomy(tax):
    if tax is None:
        return {}
    if isinstance(tax, dict):
        return tax
    try:
        return json.loads(tax)
    except (json.JSONDecodeError, TypeError):
        return {}

def get_embedding(text, model_name, sbert_model=None):
    if model_name.startswith("text-embedding"):
        response = openai.Embedding.create(
            model=model_name,
            input=text
        )
        return response['data'][0]['embedding']
    elif model_name == "sbert":
        if sbert_model is None:
            sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        return sbert_model.encode(text)
    else:
        raise ValueError("Unknown embedding model name.")
    
def compute_weighted_tax_similarity(entry, embed_model: str, sbert_model=None):
    '''
    func: calculate weighted taxonomy for values only
    input: entry: dict element of baseline results
    input: embed_model: stuff like sbert of openaiembedding
    w = [0.5,0.2,0.1,0.1,0.1]
    no source or target: return 0
    source empty for 0th level: return 0
    1 empty/impossible for both in non-0th: consider as +1.0*w[i]
    2 one empty/impossible: +0.0*w[i]
    3 +sim(val_source, val_target)*w[i]
    return 
    '''
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    weighted_tax_similarity = 0.0

    source_tax = safe_parse_taxonomy(entry.get('source_taxonomy'))
    target_tax = safe_parse_taxonomy(entry.get('target_taxonomy'))

    if not source_tax or not target_tax:
        print(f"No files: entry: {entry}")
        return 0.0

    source_items = list(source_tax.items())
    target_items = list(target_tax.items())

    if len(source_items) != 5:
        print(f"items {len(source_items)} {len(target_items)}")
        return 0.0

    for i in range(5):
        source_key, source_value = source_items[i]
        target_key, target_value = target_items[i]

        source_value = normalize_to_string(source_value)
        target_value = normalize_to_string(target_value)

        if source_value == "impossible":
            source_value = "empty"
        if target_value == "impossible":
            target_value = "empty"

        if i == 0 and source_value == "empty":
            print(f"0th level empty: entry: {entry}")
            return 0.0

        if source_value == "empty" and target_value == "empty":
            weighted_tax_similarity += weights[i]
        elif source_value == "empty" or target_value == "empty":
            weighted_tax_similarity += 0.0
        else:
            print(f"{i}th level: {source_value} {target_value}")
            source_emb = get_embedding(source_value, embed_model, sbert_model)
            target_emb = get_embedding(target_value, embed_model, sbert_model)
            sim = cosine_similarity(source_emb, target_emb)
            weighted_tax_similarity += weights[i] * sim

    return weighted_tax_similarity

def compute_dtw_seq_similarity(entry, embed_model):
    '''
    compute dynamic time warping similarity between two sequences
    '''
    seq_similarity =[]
    source_seq = safe_parse_sequence(entry.get('source_sequence'))
    target_seq = safe_parse_sequence(entry.get('target_sequence'))

    # if not source_seq or not target_seq:
    #     seq_similarity = 0.0
    # else:
    #     source_seq_str = ' '.join(source_seq)
    #     target_seq_str = ' '.join(target_seq)
    #     source_seq_emb = embed_model.encode(source_seq_str)
    #     target_seq_emb = embed_model.encode(target_seq_str)
    #     seq_similarity = cosine_similarity(source_seq_emb, target_seq_emb)    

    if not source_seq or not target_seq:
        seq_similarity = 0.0
    else:
        # Get embeddings for each step in both sequences
        emb1 = embed_model.encode(source_seq)
        emb2 = embed_model.encode(target_seq)
        distance, path = fastdtw(emb1, emb2, dist=cosine)
        # Normalize distance to similarity (lower distance = higher similarity)
        # Optional: convert to similarity score in [0, 1] range
        max_possible_distance = len(path)  # worst case = all pairs unrelated
        seq_similarity = 1 - (distance / max_possible_distance)

    return seq_similarity


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

    BASELINE_RAG_FOLDER = "/output-rag-0602/"
    BASELINE_NORAG_FOLDER = "/output-norag-0602/"
    BASELINE_RAGGOAL_FOLDER = "/output-1goalmediation-0602/"
    BASELINE_NORAGGOAL_FOLDER = "/output-1goalmediation-norag-0602/"

    SAVE_RAG_FOLDER = "/result_v6/rag.pkl"
    SAVE_NORAG_FOLDER = "/result_v6/norag.pkl"
    SAVE_RAGGOAL_FOLDER = "/result_v6/raggoal.pkl"
    SAVE_NORAGGOAL_FOLDER = "/result_v6/noraggoal.pkl"

    # Get scenegraph for source and target
    source_spatial_json_list, target_spatial_json_list, aug_levels = agent_init.get_paired_spatial_json_list(constants_init.PATH_AUGMENTATION_v6)

    source_idx_list = [i for i in range(len(source_spatial_json_list)//len(aug_levels)) for _ in range(len(aug_levels))]    

    # get evaluation results
    evaluations_rag = get_evaluations_taxseq(BASELINE_RAG_FOLDER)
    evaluations_norag = get_evaluations_taxseq(BASELINE_NORAG_FOLDER)
    evaluations_raggoal = get_evaluations_seq(BASELINE_RAGGOAL_FOLDER)
    evaluations_noraggoal = get_evaluations_seq(BASELINE_NORAGGOAL_FOLDER)

    # save as pickle
    with open(SAVE_RAG_FOLDER, 'wb') as f:
        pickle.dump(evaluations_rag, f)
    with open(SAVE_NORAG_FOLDER, 'wb') as f:
        pickle.dump(evaluations_norag, f)
    with open(SAVE_RAGGOAL_FOLDER, 'wb') as f:
        pickle.dump(evaluations_raggoal, f)
    with open(SAVE_NORAGGOAL_FOLDER, 'wb') as f:
        pickle.dump(evaluations_noraggoal, f)                        