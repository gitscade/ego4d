"""
Initialize LLM models for agents and tools
agent_funcs for making agent input
"""
'''
func: predict deep activity for source scene, using source action sequece/scene graph/RAG examples
input: (source) action sequence, scene graph
output: source deep activity
'''
import sys
import os
import subprocess
import time
import logging
from dotenv import load_dotenv
import ast
import json
import re
import pickle

#llm
from langchain_ollama import OllamaLLM
import openai
from langchain_openai.chat_models import ChatOpenAI # good for agents
from langchain_community.llms import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import f1_init.database_init as database_init


# -----------------------
# Agent Init API, LLM
# -----------------------
def check_file(path):
    try:
        with open(path, "rb") as f:
            pickle.load(f)  # try loading to ensure file is not empty or corrupted
        return True
    except (EOFError, FileNotFoundError, PermissionError, IsADirectoryError, pickle.UnpicklingError):
        return False

def SET_LLMS(api_name:str, llm_str:str, temperature:int):
    """
    input: api_name: "openai / ollama"\n
    input: llm_str: "gpt-4"\n
    output: llm_api, llm_str, llm_instance
    """
    logging.basicConfig(level=logging.ERROR)
    load_dotenv()
    parser_stroutput = StrOutputParser()

    if api_name == "openai":
        # connect to openai api key        
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # "gpt-4.1" "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"
        return api_name, llm_str, ChatOpenAI(openai_api_key=openai.api_key, model=llm_str, temperature=temperature)
    
    elif api_name == "ollama":
        # kill all existing ollama pid
        os.system("pkill -f 'ollama serve'")
        print("[INFO] Ollama 서버가 중지되었습니다.")
        try:
            output = subprocess.check_output(["pgrep", "-f", "ollama"], stderr=subprocess.DEVNULL).decode().strip()
            if output:
                print(f"[INFO] Ollama is running with PID: {output}")
            else:
                print("[WARNING] Ollama is NOT running.")
        except subprocess.CalledProcessError:
            print("[WARNING] Ollama is NOT running.")        

        # turn on new ollama server
        try:
            # Ollama 서버를 백그라운드에서 실행
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)  # 서버가 시작될 시간을 확보
            print("[INFO] Ollama server background running")
        except Exception as e:
            print(f"[ERROR] Ollama server FAIL TO RUN: {e}")       

        # "llama3.3:70b" "deepseek-r1:70b" "gemma3:27b"  "deepseek-r1:32b" 
        return api_name, llm_str, OllamaLLM(model=llm_str, temperature=temperature)
    
    return "none", "none", "none" 

def extract_all_goalstep_segments(data):
    """
    func: 
    input: data: json loaded data
    output: segments: [v1seg1, v1seg2 ... v1segn, v2seg1, ... , v2segm, .....]
    """
    segments = []
    def recurse_segments(segment, parent_id=None, level=1):
        segment_id = segment.get("number")

        
        text = json.dumps(segment.get("context"))
        metadata = {
            "level": level,
            "segment_id": segment_id,
            "parent_id": parent_id,
            "video_uid": segment.get("video_uid"),
        }
        segments.append({"text": text, "metadata": metadata})

        # Process child segments recursively 
        for child_segment in segment.get("segments", []):
            recurse_segments(child_segment, parent_id=segment_id, level=level+1)

    for video in data["videos"]:
        recurse_segments(video, parent_id=None, level=1)    
    return segments            

def extract_lower_goalstep_segments(video: dict):
    """
    func: return lev2 & lev3 segments for a single video
    input: video: one video element of json loaded file
    output: lv2segments: ['Kneads the dough with the mixer.', 'Pours the flour into the mixer', 'Organize the table', ...] -> STR: Kneads the... , Pours the flour, ...
    output: lv3segments: ['weigh the dough', 'weigh the dough', 'weigh the dough', 'move scale to tabletop', ...]
    """

    steps = []
    substeps = []

    # print(video["video_uid"])

    for i, level2_segment in enumerate(video.get("segments", [])):    
        steps.append(level2_segment["step_description"])

        for j, level3_segment in enumerate(level2_segment.get("segments", [])):
            substeps.append(level3_segment["step_description"])

    # return lv2segments, lv3segments
    steps = ", ".join(steps)
    substeps = ", ".join(substeps)
    
    # print(f"steps: {steps}")
    # print(f"susteps: {substeps}")
    if steps and substeps:
        return substeps
    elif steps:
        return steps
    else:
        return None  # or [] if you prefer an empty list as default

def extract_spatial_context(video: dict):
    """
    func: extract spatial_context section from the video dictionary
    input: video: video from which to extract spatial context
    output: json_string: [{"object_id": 1, "object_name": "oil", "init_status": {"status": "default", "container": null}}, {"object_id": 3, "object_name": "steak", "init_status": {"status": "d...
    """
    # dump =>json format (double quotes, null)
    # load =>json to python format(single quotes, None)
    scenegraph = video["spatial_data"]
    scenegraph = scenegraph = json.dumps(scenegraph)
    #print(scenegraph)
    scenegraph = json.loads(scenegraph)
    #print(scenegraph)
    return scenegraph


# -----------------------
# READ Augmented Dataset to return scene graph list
# -----------------------
def get_video_info_idxtest(source_video_idx):
    '''
    func: return all raw video info for source idx (0-70)
    return: seq, scenegraph, sea_id, scenegraph_id
    '''

def get_paired_spatial_json_list(path:str, boolauglev:bool):
    '''
    func: get augmentation data path and return scene graph list
    input: path for augmentation
    return: [source_scenegraph_json_list] [target_scenegrpah_json_list] [auglevel]
    '''
    paired_source_json_list=[]
    paired_target_json_list=[]
    auglevel=[]
    # read json files at 1st level only

    # Read source JSON files at the first nested level
    source_json_dict = {}
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if filename.endswith(".json") and os.path.isfile(file_path):
            with open(file_path, "r") as f:                
                source_data = json.load(f)
                source_uid = source_data['video_id']
                # uid=key, loaded json=data
                source_json_dict[source_uid] = source_data
    
    # numerically sort the source data
    source_json_dict = {key: source_json_dict[key] for key in sorted(source_json_dict, reverse=False)}

    # sorted keys comply with the 10 index offset for target uid in folder
    #print(list(source_json_dict.keys()))
      
    # Read folder for the target files
    for source_uid in list(source_json_dict.keys()):
        folder_path = os.path.join(path, source_uid)
        target_filenames = []
        for filename in os.listdir(folder_path):
            target_filenames.append(filename)

        #sort the filename numerically
        sorted_target_filenames = sorted(target_filenames, key=lambda f: int(re.search(r'_(\d+)\.json$', f).group(1)))

        #get auglevel(overwirtten many times but no big deal)
        if boolauglev:
            auglevel = numbers = [int(re.search(r'_(\d+)\.json$', filename).group(1)) for filename in sorted_target_filenames]
        else:
            auglevel = 1

        #read files & append both source and target list
        for filename in sorted_target_filenames:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                target_data = json.load(f)
                paired_target_json_list.append(target_data)
                paired_source_json_list.append(source_json_dict[source_uid])

                # print(f"{source_uid} {source_json_dict[source_uid]['video_id']} {target_data['video_id']} {filename}")
    # print(auglevel)
    return paired_source_json_list, paired_target_json_list, auglevel



# -----------------------
# VIDEO LIST, VECSTORE, RETRIEVER
# -----------------------
# Load VIDEO LIST (use text video list for testing)
goalstep_test_video_list = database_init.goalstep_test_video_list
spatial_test_video_list = database_init.spatial_test_video_list

# LOAD FAISS VECSTORE
goalstep_vector_store = database_init.goalstep_vector_store
spatial_vector_store = database_init.spatial_vector_store

# MAKE base:VectorStoreRetriever
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def get_video_info(source_video_idx:int):
    """
    func: return source_action_sequence, source_scene_graph
    input: source_video_idx: find in test_video_lists for goal&spatial
    ourput: source_action_sequence, source_scene_graph
    """
    source_goalstep_video = goalstep_test_video_list[source_video_idx]
    source_action_sequence = extract_lower_goalstep_segments(source_goalstep_video)
    source_action_sequence = json.dumps(source_action_sequence)
    
    source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
    source_scene_graph = extract_spatial_context(source_spatial_video)
    source_scene_graph = json.dumps(source_scene_graph, indent=2)
    return source_action_sequence, source_scene_graph

# def get_target_video_info(target_video_idx:int):
#     """
#     func: return target_action_sequence, target_scene_graph
#     input: target_video_idx: find in test_video_lists for goal&spatial
#     ourput: target_action_sequence, target_scene_graph
#     """
#     target_goalstep_video = goalstep_test_video_list[target_video_idx]
#     target_action_sequence = extract_spatial_context(target_goalstep_video)
#     target_action_sequence = json.dumps(target_action_sequence)

#     target_spatial_video = database_init.spatial_test_video_list[target_video_idx]
#     target_scene_graph = extract_spatial_context(target_spatial_video)
#     target_scene_graph = json.dumps(target_scene_graph, indent=2)
#     return target_action_sequence, target_scene_graph


if __name__=="__main__":
    #print(video["segments"][0])

    import f1_init.database_init as database_init
    source_video_idx =2
    goalstep_test_video_list = database_init.goalstep_test_video_list
    source_goalstep_video = goalstep_test_video_list[source_video_idx]
    source_sequence = extract_lower_goalstep_segments(source_goalstep_video)
    
    # print("how do you do")
    # print(source_sequence)

    # input_str = source_sequence.replace("'", '"')
    # # Now parse as JSON
    # input_json = json.loads(source_sequence)    
    #print(source_sequence)

    spatial_test_video = database_init.spatial_test_video_list[source_video_idx]
    scenegraph = extract_spatial_context(spatial_test_video)
    # print(scenegraph)
