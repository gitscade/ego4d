"""
This is folder for vector retrieval

# LOAD FAISS VECSTORE
goalstep_vector_store
spatial_vector_store

# MAKE RETRIEVER
goalstep_retriever
spatial_retriever
"""
import sys
import os
# import re
# import pickle
import langchain.retrievers
# import streamlit as st
import openai
# import pandas as pd
import logging
# import json
from dotenv import load_dotenv
#vectorstore
import langchain
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#llm
# from langchain_ollama import OllamaLLM
from langchain_community.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI # good for agents
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# #agents
# from langchain.tools import tool
# from langchain.tools import Tool
# #from langchain.agents import AgentType, initialize_agent # deprecated
# from langchain.agents import AgentType, create_react_agent, AgentExecutor
# #from langchain.memory import ConversationBufferMemory # being phased out
# from langgraph.checkpoint.memory import MemorySaver
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import script_work.agent_database as agent_database
import script_work.agent_input as agent_input
import script_work.agent_query as agent_query
import script_work.agent_prompt as agent_prompt
from util import util_constants

# -----------------------
# Path & API & Model
# -----------------------
data_path = util_constants.PATH_DATA
GOALSTEP_ANNOTATION_PATH = data_path + 'goalstep/'
SPATIAL_ANNOTATION_PATH = data_path + 'spatial/'
GOALSTEP_VECSTORE_PATH = GOALSTEP_ANNOTATION_PATH + 'goalstep_docarray_faiss'
SPATIAL_VECSTORE_PATH = SPATIAL_ANNOTATION_PATH + 'spatial_docarray_faiss'

logging.basicConfig(level=logging.ERROR)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model1 = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini") #10x cheaper
parser_stroutput = StrOutputParser()


# -----------------------
# PREPROCESS DATA
# -----------------------
# EXTRACT video list
print(GOALSTEP_ANNOTATION_PATH)
print(SPATIAL_ANNOTATION_PATH)
goalstep_videos_list = agent_database.merge_json_video_list(GOALSTEP_ANNOTATION_PATH)
spatial_videos_list = agent_database.merge_json_video_list(SPATIAL_ANNOTATION_PATH)
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
goalstep_videos_list, goalstep_test_video_list = agent_database.exclude_test_video_list(goalstep_videos_list, test_uid)
spatial_videos_list, spatial_test_video_list = agent_database.exclude_test_video_list(spatial_videos_list, test_uid)
print(f"testuid excluded: goalstep vids: {len(goalstep_videos_list)} and spatial vids: {len(spatial_videos_list)}")
print(f"testuid list: goalstep vids: {len(goalstep_test_video_list)} and spatial vids: {len(spatial_test_video_list)}")

# MAKE docu list
goalstep_document_list = agent_database.make_goalstep_document_list(goalstep_videos_list)
spatial_document = agent_database.make_spatial_document_list(spatial_videos_list)
goalstep_test_document_list = agent_database.make_goalstep_document_list(goalstep_test_video_list)
spatial_test_document_list = agent_database.make_spatial_document_list(spatial_test_video_list)

print(f"MAKE_DOCU: goalstep_document_list: {len(goalstep_document_list)}")
print(f"MAKE_DOCU: spatial_document_list: {len(spatial_document)}")
print(f"MAKE_DOCU: goalstep_document_list: {len(goalstep_test_document_list)}")
print(f"MMAKE_DOCUAKE: spatial_document_list: {len(spatial_test_document_list)}")

# -----------------------
# MAKE/LOAD FAISS Vectorstore and retrievers
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