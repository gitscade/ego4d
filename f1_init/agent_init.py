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
import logging
from dotenv import load_dotenv
import ast
import json
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
logging.basicConfig(level=logging.ERROR)
load_dotenv()
parser_stroutput = StrOutputParser()
openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL_4 = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
LLM_MODEL_4MINI = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
# LLM_MODEL_LLAMA38b = OllamaLLM(model="llama3:8b", temperature=1)
LLM_MODEL_LLAMA370b = OllamaLLM(model="llama3:70b-instruct", temperature=1)
LLM_MODEL_GEMMA327b = OllamaLLM(model="gemma3:27b", temperature=1)


# -----------------------
# Agent Input Functions
# -----------------------
import json
# TODO: def function that reads one annotation file and selects level 2 actions in formatted forms. The function should return 
"""
- read from "root/data/ego4d_annotation/demo/input"
- choose video index to use as input data
- extract goalstep lv2 segments from video
- extract spatial context from video
- parse goalstep segments
- parse spatial context segments
"""

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
    lv2segments = []
    lv3segments = []

    #print(video)
    for i, level2_segment in enumerate(video.get("segments", [])):    
        lv2segments.append(level2_segment["step_description"])

        for j, level3_segment in enumerate(level2_segment.get("segments", [])):
            lv3segments.append(level3_segment["step_description"])

    # return lv2segments, lv3segments

    lv3segments = ", ".join(lv3segments)
    return lv3segments

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
