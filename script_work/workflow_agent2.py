'''
This is an Agentic RAG for activity transfer task, based on principle of vertical activity transform
activity (source) -> common activity -> activity (target)
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


llm = OpenAI(model_name="gpt-4", temperature=0)

tool2 = Tool(name="Tool2", func=lambda x: f"Processed by tool2: {x}", description="Second tool")

agent2 = initialize_agent([tool2], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def run_agent2(input_text):
    return agent2.run(input_text)

if __name__ == "__main__":
    print(run_agent2("Test from agent2"))