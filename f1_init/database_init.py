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


if __name__ == "__main__":

    # -----------------------
    # Path & API & Model
    # -----------------------
    data_path = util_constants.PATH_DATA
    GOALSTEP_ANNOTATION_PATH = data_path + 'goalstep/'
    SPATIAL_ANNOTATION_PATH1 = data_path + 'spatial/manual'
    SPATIAL_ANNOTATION_PATH2 = data_path + 'spatial/semiauto'

    GOALSTEP_VECSTORE_PATH = GOALSTEP_ANNOTATION_PATH + 'goalstep_docarray_faiss'
    SPATIAL_VECSTORE_PATH = data_path + 'spatial/spatial_docarray_faiss'

    logging.basicConfig(level=logging.ERROR)
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # -----------------------
    # PREPROCESS DATA
    # -----------------------
    # merge individual json data into a list
    goalstep_videos_list = make_goalstep_json_video_list(GOALSTEP_ANNOTATION_PATH)
    spatial_videos_list = make_spatial_json_video_list(SPATIAL_ANNOTATION_PATH1, SPATIAL_ANNOTATION_PATH2)

    # Divide list to test list and vectorstore list
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
    goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # # IF NEEDED
    # # Make Retriever (ParentDocumentRetriever)
    # from langchain.retrievers import ParentDocumentRetriever