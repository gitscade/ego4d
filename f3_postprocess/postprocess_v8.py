import sys
import os
import openai
import logging
import json
import ast
import argparse
from dotenv import load_dotenv
import pickle
import argparse
#llm
import ollama
from langchain_ollama import OllamaLLM
# from langchain_community.llms import OpenAI
import openai
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

def get_result_list(BASELINE_FOLDER, length, isSPS=False):
    PATH_SOURCE_TARGET_OUTPUT = constants_init.PATH_SOURCE_TARGET + "/"+ BASELINE_FOLDER
    # source_spatial_json_list, target_spatial_json_list, aug_levels = agent_init.get_paired_spatial_json_list(constants_init.PATH_AUGMENTATION_v6)
    # source_idx_list = [i for i in range(len(source_spatial_json_list)//len(aug_levels)) for _ in range(len(aug_levels))]

    result_list = []
    for i in range(length):
        print(f"{i} {BASELINE_FOLDER}")
        result_dict = {}
        # -----------------------
        # CHECK PATHS
        # -----------------------
        #common
        PATH_SOURCEINFO = PATH_SOURCE_TARGET_OUTPUT + f"/pair{i}_sourceinfo.pkl" 
        PATH_TARGETINFO = PATH_SOURCE_TARGET_OUTPUT + f"/pair{i}_targetinfo.pkl" 
        PATH_AGENT1a = PATH_SOURCE_TARGET_OUTPUT + f"/pair{i}_agent1a.pkl" #core 
        PATH_AGENT3 = PATH_SOURCE_TARGET_OUTPUT + f"/pair{i}_agent3.pkl"
        PATH_AGENT1b = PATH_SOURCE_TARGET_OUTPUT + f"/pair{i}_agent1b.pkl" 
        PATH_AGENT2b = PATH_SOURCE_TARGET_OUTPUT + f"/pair{i}_agent2b.pkl"

        # -----------------------
        # GET FILES
        # -----------------------
        sourceinfo = load_file(PATH_SOURCEINFO)
        targetinfo = load_file(PATH_TARGETINFO)

        # about metadata
        idx = i
        baseline = BASELINE_FOLDER
        source_idx = sourceinfo['source_idx']      
        source_uid  = sourceinfo['source_uid']
        target_uid = targetinfo['target_uid']
        target_equal_ratio = sourceinfo['target_equal_ratio']
        trial_idx = sourceinfo['trial_index']
        source_file_name = sourceinfo['source_file_name']
        target_file_name = targetinfo['target_file_name']

        # about inputs
        source_action_sequence = sourceinfo['source_action_sequence']
        source_scene_graph = sourceinfo['source_scene_graph']
        target_scene_graph = targetinfo['target_scene_graph']
        source_goal_category = sourceinfo['source_goal_category']
        source_goal_description = sourceinfo['source_goal_description']

        #common output
        core_activity = load_file(PATH_AGENT1a)
        target_sequence = load_file(PATH_AGENT3)

        # # metadata
        result_dict["idx"]=idx
        result_dict["baseline"]=baseline
        result_dict["source_idx"]=source_idx
        result_dict["source_uid"]=source_uid
        result_dict["target_uid"]=target_uid
        result_dict["target_equal_ratio"] = target_equal_ratio
        result_dict["trial_idx"] = trial_idx
        result_dict["source_file_name"] = source_file_name
        result_dict["target_file_name"] = target_file_name

        # about inputs
        result_dict["source_action_sequence"] = source_action_sequence
        result_dict["source_scene_graph"] = source_scene_graph
        result_dict["target_scene_graph"] = target_scene_graph
        result_dict["source_goal_category"] = source_goal_category
        result_dict["source_goal_description"] = source_goal_description

        #common output
        result_dict["core_activity"]=core_activity
        result_dict["target_sequence"]=target_sequence
    
        # previous evaluator block
        result_dict["human_entitycheck_tax"]=False
        result_dict["human_entitycheck_seq"]=False   
        result_dict["human_coreactivitycheck"]=False #x
        result_dict["human_passcheck_tax"]=False
        result_dict["human_passcheck_seq"]=False

        if isSPS:
            #sps output
            source_taxonomy = load_file(PATH_AGENT1b)
            target_taxonomy = load_file(PATH_AGENT2b)

            result_dict["source_taxonomy"]=source_taxonomy
            result_dict["target_taxonomy"]=target_taxonomy   
        else:
            result_dict["source_taxonomy"]="sequence only baseline. no taxonomy"
            result_dict["target_taxonomy"]="sequence only baseline. no taxonomy"

        result_list.append(result_dict)
        # save_path = "/result_v8/"+ BASELINE_FOLDER + f"/result_{i}.pkl"
        # save_file(save_path, result_dict)
        # print(f"saved to {save_path}")

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

def save_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
            pickle.dump(data, f)

# #-----------------------------------------------------------
# # Similarity for Tax(Weight) and Sequence(DTW)
# #-----------------------------------------------------------
# def normalize_to_string(value):
#     '''
#     used so to cope with value from a dictionary that happens to be inside a list bracket    
#     '''
#     # If it's a list and the first element is a string, return that
#     if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
#         return value[0]
#     # If it's already a string, return it
#     elif isinstance(value, str):
#         return value
#     # Fallback for anything else
#     else:
#         return str(value)
    
# def cosine_similarity(vec1, vec2):
#     vec1 = np.array(vec1).reshape(1, -1)
#     vec2 = np.array(vec2).reshape(1, -1)
#     return sklearn_cosine(vec1, vec2)[0][0]

# def safe_parse_sequence(seq):
#     if seq is None:
#         return []
#     if isinstance(seq, list):
#         return seq
#     try:
#         return ast.literal_eval(seq)
#     except (ValueError, SyntaxError):
#         print("sequence safe parse value error")
#         return []

# def safe_parse_taxonomy(tax):
#     if tax is None:
#         return {}
#     if isinstance(tax, dict):
#         return tax
#     try:
#         return json.loads(tax)
#     except (json.JSONDecodeError, TypeError):
#         print("taxnomy safe parse value error")
#         return {}

# def get_embedding(text, model_name, sbert_model=None):
#     if model_name.startswith("text-embedding"):
#         response = openai.embeddings.create(
#             model=model_name,
#             input=[text]
#         )
#         return response.data[0].embedding
#     elif model_name == "sbert":
#         if sbert_model is None:
#             sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
#         return sbert_model.encode(text)
#     else:
#         raise ValueError("Unknown embedding model name.")
    
# def get_embeddings_batch(text_list, model_name): 
#     response = openai.embeddings.create(
#         input=text_list,
#         model=model_name
#     )
#     return [item.embedding for item in response.data]

# def compute_weighted_tax_similarity(entry, embed_model: str, sbert_model=None):
#     '''
#     func: calculate weighted taxonomy for values only
#     input: entry: dict element of baseline results
#     input: embed_model: stuff like sbert of openaiembedding
#     w = [0.5,0.2,0.1,0.1,0.1]
#     no source or target: return 0
#     source empty for 0th level: return 0
#     1 empty/impossible for both in non-0th: consider as +1.0*w[i]
#     2 one empty/impossible: +0.0*w[i]
#     3 +sim(val_source, val_target)*w[i]
#     return 
#     '''
#     weights = [0.5, 0.2, 0.1, 0.1, 0.1]
#     weighted_tax_similarity = 0.0

#     source_tax = safe_parse_taxonomy(entry.get('source_taxonomy'))
#     target_tax = safe_parse_taxonomy(entry.get('target_taxonomy'))

#     if not source_tax or not target_tax:
#         print(f"No files: entry: {entry}")
#         return 0.0

#     source_items = list(source_tax.items())
#     target_items = list(target_tax.items())

#     if len(source_items) != 5:
#         print(f"items {len(source_items)} {len(target_items)}")
#         return 0.0

#     for i in range(5):
#         source_key, source_value = source_items[i]
#         target_key, target_value = target_items[i]

#         source_value = normalize_to_string(source_value)
#         target_value = normalize_to_string(target_value)

#         if source_value == "impossible":
#             source_value = "empty"
#         if target_value == "impossible":
#             target_value = "empty"

#         if i == 0 and source_value == "empty":
#             print(f"0th level empty: entry: {entry}")
#             return 0.0

#         if source_value == "empty" and target_value == "empty":
#             weighted_tax_similarity += weights[i]
#         elif source_value == "empty" or target_value == "empty":
#             weighted_tax_similarity += 0.0
#         else:
#             print(f"{i}th level: {source_value} {target_value}")
#             source_emb = get_embedding(source_value, embed_model, sbert_model)
#             target_emb = get_embedding(target_value, embed_model, sbert_model)
#             sim = cosine_similarity(source_emb, target_emb)
#             weighted_tax_similarity += weights[i] * sim

#     return weighted_tax_similarity

# def compute_dtw_seq_similarity(entry, embed_model: str):
#     '''
#     Compute dynamic time warping similarity between two sequences.
#     '''
#     source_seq = safe_parse_sequence(entry.get('source_sequence'))
#     target_seq = safe_parse_sequence(entry.get('target_sequence'))

#     if not source_seq or not target_seq:
#         return 0.0

#     # Get embeddings for each step in sequence
#     if "text-embedding" in embed_model:
#         emb1 = get_embeddings_batch(source_seq, model_name=embed_model)
#         emb2 = get_embeddings_batch(target_seq, model_name=embed_model)
#     elif "sbert" in embed_model:
#         sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
#         emb1 = sbert_model.encode(source_seq)
#         emb2 = sbert_model.encode(target_seq)
#     else:
#         raise ValueError(f"Unsupported embedding model: {embed_model}")

#     # Now compute DTW similarity regardless of model type
#     distance, path = fastdtw(emb1, emb2, dist=cosine)
#     max_possible_distance = len(path)
#     seq_similarity = 1 - (distance / max_possible_distance)

#     return seq_similarity




if __name__ == "__main__":
    # -----------------------
    # ARGPARSE: cd f4_evaluate    python evaluate_evaluator_v8.py 
    # -----------------------        
    parser = argparse.ArgumentParser(description="baseline(idx 0-4), temperature(0-1.0)")
    parser.add_argument("-b", "--baseline", dest="baseline_idx", default=0)
    parser.add_argument("-t", "--temperature", dest="temperature_value", default=0.5)
    args = parser.parse_args()

    # -----------------------
    # Evaluator
    # -----------------------
    tool_api_name = "openai"
    tool_model_name = "gpt-4.1mini" #"gpt-4.1"
    openai.api_key = os.getenv("OPENAI_API_KEY")  

    TOOL_LLM_API, TOOL_LLM_STR, TOOL_LLM_CHAT = agent_init.SET_LLMS(tool_api_name, tool_model_name, temperature=0.2)    

    BASELINE_FOLDERS = [
        "output0_1",
        "output1-norag_1",
        "output1-rag_1",
        "output2-norag_1",
        "output2-rag_1",
    ]

    POSTPROCESS_PATH = os.getcwd() + "/f3_postprocess"
    EVAL_RESULT_FOLDERS = [
        POSTPROCESS_PATH + "/result_v8/base0_1.pkl",
        POSTPROCESS_PATH + "/result_v8/base1-norag_1.pkl",
        POSTPROCESS_PATH + "/result_v8/base1-rag_1.pkl",
        POSTPROCESS_PATH + "/result_v8/base2-norag_1.pkl",
        POSTPROCESS_PATH + "/result_v8/base2-rag_1.pkl",
    ]

    # get evaluation results
    base0_eval_result = get_result_list(BASELINE_FOLDERS[0], 600)
    base1_norag_eval_result = get_result_list(BASELINE_FOLDERS[1], 600)
    base1_rag_eval_result = get_result_list(BASELINE_FOLDERS[2], 600)
    base2_norag_eval_result = get_result_list(BASELINE_FOLDERS[3], 600, isSPS=True)
    base2_rag_eval_result = get_result_list(BASELINE_FOLDERS[4], 600, isSPS=True)

    # save as pickle
    with open(EVAL_RESULT_FOLDERS[0], 'wb') as f:
        pickle.dump(base0_eval_result, f)
    with open(EVAL_RESULT_FOLDERS[1], 'wb') as f:
        pickle.dump(base1_norag_eval_result, f)        
    with open(EVAL_RESULT_FOLDERS[2], 'wb') as f:
        pickle.dump(base1_rag_eval_result, f)
    with open(EVAL_RESULT_FOLDERS[3], 'wb') as f:
        pickle.dump(base2_norag_eval_result, f)
    with open(EVAL_RESULT_FOLDERS[4], 'wb') as f:
        pickle.dump(base2_rag_eval_result, f)                        
                 