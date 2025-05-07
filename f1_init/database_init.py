# THIS SCRIPT MAKES VECTORSTORE.
# SAME SCRIPT IS TO BE MADE IN f01_inti folder, so if substitute script is ready, delete this script



"""
This is folder for vector retrieval
# MAKE/SAVE FAISS VECSTORE

# LOAD FAISS VECSTORE
single document chunk vecstore: goalstep_vector_store, spatial_vector_store

# MAKE RETRIEVER
VectorstoreRetriver: goalstep_retriever, spatial_retriever
ParentDocumentRetriver: NOT YET
"""
import sys
import os
import openai
import logging
import json
from dotenv import load_dotenv
#vectorstore
from langchain_community.vectorstores import FAISS
#llm
from langchain_openai.chat_models import ChatOpenAI # good for agents
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import f1_init.agent_database as agent_database
from util import util_constants

# -----------------------
# Path & API & Model
# -----------------------
data_path = util_constants.PATH_DATA
GOALSTEP_ANNOTATION_PATH = data_path + 'goalstep/'
SPATIAL_ANNOTATION_PATH = data_path + 'spatial/'
SPATIAL_ANNOTATION_V1_PATH = data_path + 'linked_result_v1'
GOALSTEP_VECSTORE_PATH = GOALSTEP_ANNOTATION_PATH + 'goalstep_docarray_faiss'
SPATIAL_VECSTORE_PATH = SPATIAL_ANNOTATION_PATH + 'spatial_docarray_faiss'

logging.basicConfig(level=logging.ERROR)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model1 = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini") #10x cheaper
parser_stroutput = StrOutputParser()



# -----------------------
# DATABASE FUNCS
# -----------------------
def merge_json_video_list(path):
    """
    func: read all json files in directory & merge "videos" list of all files into one
    input: path to the directory
    output: merged_vides = []
    """
    merged_videos = []

    for filename in os.listdir(path):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)

            #open json and read
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    if 'videos' in data:
                        merged_videos.extend(data["videos"])
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")
    return merged_videos

def merge_json_video_list_exclude_files_in_path2(path, path2):
    """
    func: read all json files in directory & merge "videos" list of all files into one
    func: exclude file with same names found in path2
    input: path to the directory
    output: merged_vides = []
    """
    merged_videos = []
    filenames_path2 = [f for f in os.listdir(path2) if f.endswith(".json")]

    for filename in os.listdir(path):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)

            #check for same file and pass if there is one
            if filename in filenames_path2:
                print(f'passing duplicate file {filename}')
                continue
            else:
                print(filename)

            #open json and read
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    if 'videos' in data:
                        merged_videos.extend(data["videos"])
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")
    return merged_videos


def exclude_test_video_list(video_list, exclude_uid_list):
    """
    func: exclude video element that contains uid element of the exclude_uid_list
    func: reorder test_video_list, so that it is sorted like the exclude_uid_list
    input: video_list, exclude_uid_list(uid_list)
    output: new_video_list, test_video_list
    """
    new_video_list = []
    test_video_list = []    
    for video in video_list:
        uid = video["video_uid"]
        if uid in exclude_uid_list:
            test_video_list.append(video)
            # print(uid)
        else:
            new_video_list.append(video)

    # sort test video list so that it follows sort order of uid_list
    test_video_list = sorted(test_video_list, key=lambda d: exclude_uid_list.index(d["video_uid"]))
    # for video in test_video_list:
    #     print(video["video_uid"])
    
    return new_video_list, test_video_list


def make_goalstep_document_list(video_list):
    """
    func: return document list for making vectorstore
    input: video_list
    output: list of documents, where each document is a single segment
    """
    document_list = []
    # Traverse the JSON structure
    for video in video_list:
        # Create a document for the video
        video_doc = Document(
            page_content=f"Video UID: {video['video_uid']}\nGoal: {video['goal_description']}",
            metadata={
                "type": "video",
                "video_uid": video["video_uid"],
                "goal_category": video["goal_category"],
                "goal_description": video["goal_description"],
                "start_time": video["start_time"],
                "end_time": video["end_time"],
            },
        )
        document_list.append(video_doc)

        # Traverse level 2 segments and CREATE DOCUMENT and APPEND
        for i, level2_segment in enumerate(video.get("segments", [])):
            level2_doc = Document(
                page_content=f"Level 2 Segment {i + 1} for Video {video['video_uid']}\nStep: {level2_segment['step_description']}",
                metadata={
                    "type": "level2",
                    "video_uid": video["video_uid"],
                    "start_time": level2_segment["start_time"],
                    "end_time": level2_segment["end_time"],
                    "step_category": level2_segment["step_category"],
                    "step_description": level2_segment["step_description"],
                },
            )
            document_list.append(level2_doc)

            # Traverse level 3 segments and CREATE DOCUMENT and APPEND
            for j, level3_segment in enumerate(level2_segment.get("segments", [])):
                level3_doc = Document(
                    page_content=f"Level 3 Segment {j + 1} for Level 2 Segment {i + 1} in Video {video['video_uid']}\nStep: {level3_segment['step_description']}",
                    metadata={
                        "type": "level3",
                        "video_uid": video["video_uid"],
                        "parent_level1_start_time": level2_segment["start_time"],
                        "start_time": level3_segment["start_time"],
                        "end_time": level3_segment["end_time"],
                        "step_category": level3_segment["step_category"],
                        "step_description": level3_segment["step_description"],
                    },
                )
                document_list.append(level3_doc)

    # Output the documents
    # for doc in document_list:
    #     print("Page Content:", doc.page_content)
    #     print("Metadata:", doc.metadata)
    #     print("-" * 50)
    return document_list



def make_spatial_document_list(video_list):
    """
    func: return document list with parent info and lv info for spatial annotation files
    input: video_list for spatial annotation
    output: list of documents for spatial annotation
    """
    document_list = []
    # Traverse the JSON structure
    for video in video_list:
        # Create a document for the video(level1)
        video_doc = Document(
            page_content=f"Video UID: {video['video_uid']}\nGoal: {video['goal_description']}\nSpatial_context: {video['spatial_context']}",
            metadata={
                "type": "level1",
                "video_uid": video["video_uid"],
                "goal_category": video["goal_category"],
                "goal_description": video["goal_description"],        
            },
        )
        document_list.append(video_doc)
        
        # Traverse level 2 segments and CREATE DOCUMENT and APPEND
        for i, level2_segment in enumerate(video.get("segments", [])):
            level2_doc = Document(
                page_content=f"Level 2 Segment {i + 1} for level 1 {video['video_uid']}\nContext: {level2_segment['context']}",
                metadata={
                    "type": "level2",
                    "video_uid": video["video_uid"],
                    "number": level2_segment["number"],
                    "level": level2_segment["level"],
                },
            )
            document_list.append(level2_doc)

            # Traverse level 3 segments and CREATE DOCUMENT and APPEND
            for j, level3_segment in enumerate(level2_segment.get("segments", [])):
                level3_doc = Document(
                    page_content=f"Level 3 Segment {j + 1} for Level 2 Segment {i + 1} in Video {video['video_uid']}\nContext: {level3_segment['context']}",
                    metadata={
                        "type": "level3",
                        "video_uid": video["video_uid"],
                        "number": level3_segment["number"],
                        "level": level3_segment["level"],
                    },
                )
                document_list.append(level3_doc)

    # Output the documents
    # for doc in document_list:
    #     print("Page Content:", doc.page_content)
    #     print("Metadata:", doc.metadata)
    #     print("-" * 50)
    return document_list




# -----------------------
# PREPROCESS DATA
# -----------------------
# EXTRACT video list
print(GOALSTEP_ANNOTATION_PATH)
print(SPATIAL_ANNOTATION_PATH)
goalstep_videos_list = merge_json_video_list(GOALSTEP_ANNOTATION_PATH)
spatial_videos_list = merge_json_video_list(SPATIAL_ANNOTATION_PATH)
spatial_videos_list2 = merge_json_video_list_exclude_files_in_path2(SPATIAL_ANNOTATION_V1_PATH, SPATIAL_ANNOTATION_PATH)
print(f"spatial vids1: {len(spatial_videos_list)}")
print(f"spatial vids2: {len(spatial_videos_list2)}")
print(f"goalstep vids: {len(goalstep_videos_list)} and spatial vids: {len(spatial_videos_list)}")

# EXCLUDE test videos
test_uid = [
    "dcd09fa4-afe2-4a0d-9703-83af2867ebd3", #make potato soap
    "46e07357-6946-4ff0-ba36-ae11840bdc39", #make tortila soap
    "026dac2d-2ab3-4f9c-9e1d-6198db4fb080", #prepare steak
    "2f46d1e6-2a85-4d46-b955-10c2eded661c", #make steak
    "14bcb17c-f70a-41d5-b10d-294388084dfc", #prepare garlic(peeling done)
    "487d752c-6e22-43e3-9c08-627bc2a6c6d4", #peel garlic
    "543e4c99-5d9f-407d-be75-c397d633fe56", #make sandwich
    "24ba7993-7fc8-4447-afd5-7ff6d548b11a", #prepare sandwich bread
    "e09a667f-04bc-49b5-8246-daf248a29174", #prepare coffee
    "b17ff269-ec2d-4ad8-88aa-b00b75921427", #prepare coffee and bread
    "58b2a4a4-b721-4753-bfc3-478cdb5bd1a8" #prepare tea and pie
]
goalstep_videos_list, goalstep_test_video_list = exclude_test_video_list(goalstep_videos_list, test_uid)
spatial_videos_list, spatial_test_video_list = exclude_test_video_list(spatial_videos_list, test_uid)
print(f"testuid excluded: goalstep vids: {len(goalstep_videos_list)} and spatial vids: {len(spatial_videos_list)}")
print(f"testuid list: goalstep vids: {len(goalstep_test_video_list)} and spatial vids: {len(spatial_test_video_list)}")

# MAKE docu list
goalstep_document_list = make_goalstep_document_list(goalstep_videos_list)
spatial_document = make_spatial_document_list(spatial_videos_list)
goalstep_test_document_list = make_goalstep_document_list(goalstep_test_video_list)
spatial_test_document_list = make_spatial_document_list(spatial_test_video_list)

print(f"MAKE_DOCU: goalstep_document_list: {len(goalstep_document_list)}")
print(f"MAKE_DOCU: spatial_document_list: {len(spatial_document)}")
print(f"MAKE_DOCU: goalstep_document_list: {len(goalstep_test_document_list)}")
print(f"MMAKE_DOCUAKE: spatial_document_list: {len(spatial_test_document_list)}")

# -----------------------
# VIDEO LIST, VECSTORE, RETRIEVER
# -----------------------
# ONE document = one chunk for now
embeddings = OpenAIEmbeddings()
if not os.path.exists(GOALSTEP_VECSTORE_PATH + '/index.faiss'):
    print(f"MAKE FAISS GOALSTEP: {GOALSTEP_VECSTORE_PATH}")
    goalstep_vector_store =  FAISS.from_documents(goalstep_document_list, embeddings)
    goalstep_vector_store.save_local(GOALSTEP_VECSTORE_PATH)
else:
    print(f"LOAD FAISS GOALSTEP: {GOALSTEP_VECSTORE_PATH}")

if not os.path.exists(SPATIAL_VECSTORE_PATH + '/index.faiss'):
    print(f"MAKE FAISS SPATIAL: {SPATIAL_VECSTORE_PATH}")
    spatial_vector_store = FAISS.from_documents(spatial_document, embeddings)
    spatial_vector_store.save_local(SPATIAL_VECSTORE_PATH)
else:
    print(f"LOAD FAISS SPATIAL: {SPATIAL_VECSTORE_PATH}")

# LOAD FAISS VECSTORE
goalstep_vector_store = FAISS.load_local(GOALSTEP_VECSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
spatial_vector_store = FAISS.load_local(SPATIAL_VECSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

# MAKE RETRIEVER (Makes Vectorstore retriever)
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # IF NEEDED
# # Make Retriever (ParentDocumentRetriever)
# from langchain.retrievers import ParentDocumentRetriever