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

from langchain.llms import OpenAI # good for single return task
from langchain_openai.chat_models import ChatOpenAI # good for agents

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent

# Define LLM
llm = OpenAI(model_name="gpt-4", temperature=0)

# Define tool for agent1
tool1 = Tool(name="Tool1", func=lambda x: f"Processed by tool1: {x}", description="First tool")

# Initialize agent1
agent1 = initialize_agent([tool1], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def run_agent1(input_text):
    return agent1.run(input_text)

if __name__ == "__main__":
    print(run_agent1("Start process"))