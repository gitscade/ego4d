# =============================================
# Databse
# make vectorstore based on 
# =============================================
import os
import json
import pickle
import re
from langchain.schema import Document
from langchain_community.vectorstores import DocArrayInMemorySearch

# TODO: draw diagram for the input_soruce script
# TODO: read annotationfiles and kind of merge them?
# TODO: read original data
# TODO: just use document listing to make two vectorstores
# TODO: implement functions to 



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

        # Traverse level 1 segments and CREATE DOCUMENT and APPEND
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
        # Create a document for the video
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
        
        # Traverse level 1 segments and CREATE DOCUMENT and APPEND
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

            # Traverse level 2 segments and CREATE DOCUMENT and APPEND
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


def make_save_vectorstore(input_list, embeddings, isDocument):
    """
    func: make vectorstore with json files
    input: a single list named videos = []
    output: return and save vecstore
    """
    vectorstore = []
    if isDocument:
        vectorstore = DocArrayInMemorySearch.from_documents(input_list, embedding=embeddings)
    else:
        vectorstore = DocArrayInMemorySearch.from_texts(input_list, embedding=embeddings)

    return vectorstore