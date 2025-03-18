'''
This is an Agentic RAG for recognition of activity based on input action sequence and spatial context
'''
import sys
import os
import re
import pickle
import streamlit as st
import openai
import pandas as pd
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory


