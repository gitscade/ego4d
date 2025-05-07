"""
Initialize LLM models for agents and tools
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

logging.basicConfig(level=logging.ERROR)
load_dotenv()
parser_stroutput = StrOutputParser()

openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL_4 = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
LLM_MODEL_4MINI = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
# LLM_MODEL_LLAMA38b = OllamaLLM(model="llama3:8b", temperature=1)
LLM_MODEL_LLAMA370b = OllamaLLM(model="llama3:70b-instruct", temperature=1)
LLM_MODEL_GEMMA327b = OllamaLLM(model="gemma3:27b", temperature=1)