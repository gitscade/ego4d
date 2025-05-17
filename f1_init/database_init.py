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
from langchain.schema import Document
from dotenv import load_dotenv
#vectorstore
from langchain_community.vectorstores import FAISS
#llm
# from langchain_openai.chat_models import ChatOpenAI # good for agents
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_core.output_parsers import StrOutputParser
#custom packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
from util import util_constants

# -----------------------
# Pick out annotation
# -----------------------
def make_spatial_json_video_uid_list(video_list):
    """
    func: read all json files and return its video_uid as a list
    input: video_list
    output: goalstep_video_uid_list
    """
    goalstep_video_uid_list = []
    for video in video_list:
        uid = video["video_id"]
        goalstep_video_uid_list.append(uid)
    
    return goalstep_video_uid_list

def exclude_test_video_list(video_list, exclude_uid_list, key_name: str):
    """
    func: exclude video element that contains uid element of the exclude_uid_list
    func: reorder test_video_list, so that it is sorted like the exclude_uid_list
    input: video_list, exclude_uid_list(uid_list)
    output: new_video_list, test_video_list
    """
    new_video_list = []
    test_video_list = []    
    for video in video_list:
        uid = video[key_name]
        if uid in exclude_uid_list:
            test_video_list.append(video)
            # print(uid)
        else:
            new_video_list.append(video)

    # sort test video list so that it follows sort order of uid_list
    test_video_list = sorted(test_video_list, key=lambda d: exclude_uid_list.index(d[key_name]))
    # for video in test_video_list:
    #     print(video["video_uid"])
    
    return new_video_list, test_video_list


# -----------------------
# DATABASE FUNCS
# -----------------------
def make_goalstep_json_video_list(path):
    """
    func: read all json files in directory & merge "videos" list of all files into one
    input: path to the directory
    output: merged_vides = []
    """
    goalstep_video_list = []

    for filename in os.listdir(path):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)

            #open json and read
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    if 'videos' in data:
                        goalstep_video_list.extend(data["videos"])
                except json.JSONDecodeError as e:
                    print(f"Error reading goalstep {filename}: {e}")
    return goalstep_video_list

def make_spatial_json_video_list(path1, path2):
    """
    *spatial videos are all individual dict and there is no "videos" key.
    func: read all json in dir recursively & merge all individual dicts to one list
    input: path to manual annotation, path to semiauto annotation
    output: merged_videos = []
    """
    spatial_video_list = []

    # manual annotation merge
    for filename in os.listdir(path1):
        if filename.endswith('.json'):
            file_path = os.path.join(path1, filename)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    spatial_video_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error reading spatial {filename}: {e}")

    # semiauto annotation merge
    for filename in os.listdir(path2):
        if filename.endswith('.json'):
            file_path = os.path.join(path2, filename)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    spatial_video_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error reading spatial {filename}: {e}")      

    return spatial_video_list

def make_spatial_json_video_list_singlepath(path1):
    """
    *spatial videos are all individual dict and there is no "videos" key.
    func: read all json in dir recursively & merge all individual dicts to one list
    input: path to manual annotation, path to semiauto annotation
    output: merged_videos = []
    """
    spatial_video_list = []

    # manual annotation merge
    for filename in os.listdir(path1):
        if filename.endswith('.json'):
            file_path = os.path.join(path1, filename)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    spatial_video_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error reading spatial {filename}: {e}")

    return spatial_video_list



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

def exclude_test_video_list(video_list, exclude_uid_list, key_name: str):
    """
    func: exclude video element that contains uid element of the exclude_uid_list
    func: reorder test_video_list, so that it is sorted like the exclude_uid_list
    input: video_list, exclude_uid_list(uid_list)
    output: new_video_list, test_video_list
    """
    new_video_list = []
    test_video_list = []    
    for video in video_list:
        uid = video[key_name]
        if uid in exclude_uid_list:
            test_video_list.append(video)
            # print(uid)
        else:
            new_video_list.append(video)

    # sort test video list so that it follows sort order of uid_list
    test_video_list = sorted(test_video_list, key=lambda d: exclude_uid_list.index(d[key_name]))
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
    func: return document list with "spatial_data(initial_graph)", and "segments(lowest level actions)"
    input: video_list for spatial annotation
    output: list of documents for spatial annotation
    """
    document_list = []
    # Traverse the JSON structure
    for video in video_list:
        # Create a document for the video(level1)
        video_doc = Document(
            page_content=f"Video UID: {video['video_id']}\nGoal: {video['goal_description']}\ninitial_state: {video['spatial_data']}",

            # metadata is used if filter is applied in retriever
            metadata={
                "type": "initial_state",
                "video_uid": video["video_id"],   
            },
        )
        document_list.append(video_doc)
        
        # CREATE DOCUMENT for each lev2 step in segment and append
        for i, segment in enumerate(video.get("segments", [])):
            step = Document(
                page_content=f"Video UID: {video['video_id']}\nGoal: {video['goal_description']}\step: {segment['context']}",

                # metadata is used if filter is applied in retriever
                metadata={
                    "type": "step",
                    "video_uid": video["video_id"],
                    "step_description": segment["description"]                        
                },
            )
            document_list.append(step)
    return document_list

# -----------------------
# Path & API & Model
# -----------------------
data_path = util_constants.PATH_DATA
GOALSTEP_ANNOTATION_PATH = data_path + 'goalstep/'
SPATIAL_ANNOTATION_PATH_MANUAL = data_path + 'spatial_all/manual'
SPATIAL_ANNOTATION_PATH_SEMI = data_path + 'spatial_all/semi'
GOALSTEP_VECSTORE_PATH = data_path + 'goalstep_docarray_faiss'
SPATIAL_VECSTORE_PATH = data_path + 'spatial_docarray_faiss'

# print(GOALSTEP_ANNOTATION_PATH)
# print(SPATIAL_ANNOTATION_PATH_MANUAL)
# print(SPATIAL_ANNOTATION_PATH_SEMI)
# print(GOALSTEP_VECSTORE_PATH)
# print(SPATIAL_VECSTORE_PATH)

# -----------------------
# PREPARE VIDEOLIST, DOCUMENTLIST
# -----------------------
# merge individual json data into a list
goalstep_videos_list = make_goalstep_json_video_list(GOALSTEP_ANNOTATION_PATH)
spatial_videos_list = make_spatial_json_video_list(SPATIAL_ANNOTATION_PATH_MANUAL, SPATIAL_ANNOTATION_PATH_SEMI)
print(f"all: goalstep vids: {len(goalstep_videos_list)}")
print(f"all: spatial vids: {len(spatial_videos_list)}")

# Divide list to test list and vectorstore list
test_uid = [
    #11 manual
    "dcd09fa4-afe2-4a0d-9703-83af2867ebd3", #make potato soap
    "46e07357-6946-4ff0-ba36-ae11840bdc39", #make tortila soap
    "026dac2d-2ab3-4f9c-9e1d-6198db4fb080", #prepare steak
    "2f46d1e6-2a85-4d46-b955-10c2eded661c", #make steak , prepare meal
    "14bcb17c-f70a-41d5-b10d-294388084dfc", #prepare garlic(peeling done), prepare garlic butter
    "487d752c-6e22-43e3-9c08-627bc2a6c6d4", #peel garlic, prepare dish
    "543e4c99-5d9f-407d-be75-c397d633fe56", #make sandwich, prepare sandwich
    "24ba7993-7fc8-4447-afd5-7ff6d548b11a", #prepare sandwich bread, make sandwich
    "e09a667f-04bc-49b5-8246-daf248a29174", #prepare coffee, organize groceries??
    "b17ff269-ec2d-4ad8-88aa-b00b75921427", #prepare coffee and bread, prepare breakfast
    "58b2a4a4-b721-4753-bfc3-478cdb5bd1a8", #prepare tea and pie, serve pie?
    #11 manual
    "28e0affc-cacb-4db8-ab32-dfc16931b86a",
    "e72082e8-f9e6-42ac-ac45-de30f9adee9d",
    "f0204f83-ea03-4c33-b7e7-13d2188ab3e5",
    "9fabfbc8-1d5c-495e-9bb2-03795f0145ae",
    "01ce4fd6-197a-4792-8778-775b03780369",
    "47bb1fd4-d41f-42b4-8d0c-29c4e9fdff9f",
    "7e8d03f2-2ff9-431d-af81-e5ffcd954a63",
    "89857b33-fa50-469a-bbb3-91c8ab655931",
    "5c2e910c-84e0-4042-b5d6-880a731c3e67",
    "737e9619-7768-407c-8a4f-6fe1e8d61f04",
    "abab0e69-f7e4-40c1-aa58-375798df487a",
    #task
    "02a06bf1-51b8-4902-b088-573e29fcd7ec",
    "1a894d3c-b3ef-448a-a3de-2b38677cef36",
    "1dc85adb-fbdd-4275-b9cf-42976acb4d14",
    "2978ddbc-cdc9-4bfa-9a7c-4bf056904010",
    "29e00040-6e0f-4f0e-816d-1ac97c1e5485",
    "2ba0becb-58c2-43a1-97bb-7e153a34eb47",
    "2bc7d6fa-a02e-4367-b316-d6b4e8a2ce3f",
    "2c27b5f1-4af6-49ad-a43c-3efb0c150868",
    "2da5c1ee-bd40-406d-83a7-2f3d93293949",
    "31d6fe77-da70-42da-8f47-66bb79b9285b",
    "321b5e21-2951-40c9-a2f9-6ce0c145cfb8",
    "341b5211-bb72-4bec-bd3d-c0d518887960",
    "35080724-6604-401c-8b06-19b7cece3d45",
    "3728f856-0d47-4614-824f-37b6dda8e357",
    "38a7b760-56f9-4565-8b70-f8dad5768ace",
    "3ec3eab7-842d-409d-8866-42ddcbd24cd9",
    "4fa75795-ddc4-4582-9715-bb7887439263",
    "5461912b-69cd-40d7-8f79-50832f92f049",
    "56fe0c73-77c4-40d9-a687-b2df28d5f7d7",
    "5c15607b-96af-4503-84b4-d1745f3a3ae0",
    "6628a2fb-19e2-4fe5-aedb-92fe5ceee9c9",
    "690f58f1-f18c-4415-bab0-787c2f83d051",
    "6ac1d2ed-1f6b-4828-a1ab-f81c40bd5e80",
    "6dafeac7-75b6-4d69-96f7-d08708a0a99e",
    "748536e4-636a-4dc6-b1a7-d9cbfdc1cffd",
    "892629b0-61eb-425d-97f4-7d213074c435",
    "907fd0e7-6821-4e2d-9c62-6d7afad5a9d1",
    "98434f4c-6216-4067-ad59-4a89cb47bb9b",
    "a267b011-b1db-4e3c-aa49-438e2afdd6dc",
    "a6419de9-1e40-4793-b21b-9c8d9038835a",
    "ab7ed4f7-10ee-4ccb-bb21-4853c9018b1e",
    "ae2d99c2-1720-4354-bc4d-f7bc3e4ee28d",
    "b4072935-56a6-4765-bb4d-d5f6bbeb95b9",
    "b83285c5-0b88-4ced-a52e-5c34ea371507",
    "cf95d6a4-6ad7-462c-9700-9f04bd993667",
    "d7a2e92e-dc74-4e79-be04-a86f829fc3ec",
    "daf5384b-ea5c-4cce-bb8a-540a360075bf",
    "debfb68a-eae2-464e-847a-cd3fea23f3ca",
    "e250017c-16ff-4825-9c30-160f391e1549",
    "e4ad6fd7-2e3e-4991-b392-a0056f702286",
    "e6231d1a-1f7f-4198-a499-7635509adfaf",
    "ec3556de-be79-4ad4-aa0f-eaca48abb5d5",
    "ed60dcdb-b273-44e7-b5dc-f9527d7c403f",
    "edc1869c-8a97-44fd-ab47-63fda4a54df9",
    "f5ac654b-8f39-427b-856f-4a9a2d4a3020",
    "fea524d4-a1b6-466c-ac48-8777c3fd173d",
    "grp-690f58f1-f18c-4415-bab0-787c2f83d051",
    "grp-b59f7f5d-2991-49a6-8e88-0e2f2db92585",
    "grp-ffd863cb-f06b-404e-a013-54acb61f1ed9",
]
goalstep_videos_list, goalstep_test_video_list = exclude_test_video_list(goalstep_videos_list, test_uid, 'video_uid')
spatial_videos_list, spatial_test_video_list = exclude_test_video_list(spatial_videos_list, test_uid, 'video_id')
print(f"testuid excluded: goalstep vids: {len(goalstep_videos_list)}")
print(f"testuid excluded: spatial vids: {len(spatial_videos_list)}")
print(f"testuid list: test goalstep vids: {len(goalstep_test_video_list)}")
print(f"testuid list: test spatial vids: {len(spatial_test_video_list)}")

# MAKE docu list
goalstep_document_list = make_goalstep_document_list(goalstep_videos_list)
goalstep_test_document_list = make_goalstep_document_list(goalstep_test_video_list)
spatial_document = make_spatial_document_list(spatial_videos_list)
spatial_test_document_list = make_spatial_document_list(spatial_test_video_list)
print(f"MAKE_DOCU: goalstep_document_list: {len(goalstep_document_list)}")
print(f"MAKE_DOCU: goalstep_document_list: {len(goalstep_test_document_list)}")
print(f"MAKE_DOCU: spatial_document_list: {len(spatial_document)}")
print(f"MAKE_DOCUAKE: spatial_document_list: {len(spatial_test_document_list)}")


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

# MAKE Vectorstore RETRIEVER (retriever looks at page_content of documents. metadata is used for manual filtering.)
# goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # IF NEEDED
# # Make Retriever (ParentDocumentRetriever)
# from langchain.retrievers import ParentDocumentRetriever