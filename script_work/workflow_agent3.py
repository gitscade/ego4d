'''
This is an Agentic RAG for prediction of action sequence, based on given activity
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
#vectorstore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#llm
from langchain_ollama import OllamaLLM
from langchain_community.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI # good for agents
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#agents
from langchain.tools import tool
from langchain.tools import Tool
#from langchain.agents import AgentType, initialize_agent # deprecated
from langchain.agents import AgentType, create_react_agent, AgentExecutor
#from langchain.memory import ConversationBufferMemory # being phased out
from langgraph.checkpoint.memory import MemorySaver
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import script_work.agent_database as agent_database
import script_work.agent_input as agent_input
import script_work.agent_query as agent_query
import script_work.agent_prompt as agent_prompt
from util import util_constants


# -----------------------
# CONFIGURATION
# -----------------------
ALLOWED_WORDS = {
    "nouns": {"apple", "banana", "computer", "data", "research"},
    "verbs": {"calculate", "analyze", "study", "process"}
}

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

# MAKE RETRIEVER
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# -----------------------
# TOOL FUNCTION
# -----------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    return goalstep_vector_store.invoke(query)

def spatial_information_retriver(query:str):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    return spatial_vector_store.invoke(query)

# TODO2:
# fetching whole documents based on retrieval system.
# 

def action_sequence_generation(query: str, goalstep_info_example: str, spatial_info_example: str):
    prompt = f"You are an action sequence generator that breaks down activity in the query into step by step action. The activity is performed only with the entities of various states given by the query's spatial_information.\n You can also retrieve goalstep information in other environments using goalstep_retriever to see which activity might lead to which specific actions.\n You can also retrieve spatial information for other environments to see how entities and their state changes in other environment takes place.\n"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an action sequence generator"},
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    
    return response["choices"][0]["message"]["content"]


def action_sequence_validation(query: str, generated_action_sequence: str):
    prompt = f"You are an action sequence validator. You have to check three things. First, you need to see whether the current action sequence fulfills the target activity given in the query. Second, you need to see whether the actions are performable with entities of various states given in the query as spatial_information. Third, you need to check if the sequence of actions are performed in logical step. When any of the three items in the checklist fails, you have to generate reasons why validation failed to the action_sequence_generation tool for regenerating answer.\n"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an action sequence validator"},
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    
    return response["choices"][0]["message"]["content"]


# -----------------------
# DEFINE TOOLS with TOOLFUNC
# -----------------------
goalstep_tool_obj = Tool(
    name = "goalstep_retriever_tool",
    func = goalstep_information_retriever,
    description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
)

spatial_tool_obj = Tool(
    name = "spatial_retriever_tool",
    func = spatial_information_retriver,
    description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
)

sequence_generation_tool_obj = Tool(
    name = "action_sequence generation tool",
    func = action_sequence_generation,
    description = "Action sequence generation tool, which can break down the given activity into smaller actions. Additional information on the current target action, target environment is needed. Additional examples from other environments can also be used. Input: query(str). Output: action_sequence_steps(str)"
)

sequence_validation_tool_obj = Tool(
    name = "action_sequence validation tool",
    func = action_sequence_validation,
    description = "Input: query(str), action_sequence(str). Output: command to call action_sequence_generation_tool_obj again if validation fails. If validation passes, print out the input action_sequence(str)."
)

# factorial_tool_obj = Tool(
#     name="factorial_tool",
#     func=factorial_tool,
#     description="Calculates factorial. Input: number(int). Output: factorial(int)."
# )
# @tool
# def check_answer_relevance(question: str, answer: str) -> str:
#     """Check if the provided answer correctly and fully addresses the given question."""
#     prompt = f"Does this answer correctly and fully respond to the question?\n\n"
#     prompt += f"Question: {question}\nAnswer: {answer}\n"
#     prompt += "Reply with 'Yes' or 'No' and explain why."
#     return LLM_MODEL.invoke(prompt)

# @tool
# def enforce_lexical_constraints(answer: str) -> str:
#     """Ensure the answer only contains approved nouns and verbs."""
#     words = set(re.findall(r'\b\w+\b', answer.lower()))
#     invalid_words = words - ALLOWED_WORDS["nouns"] - ALLOWED_WORDS["verbs"]
#     if invalid_words:
#         return f"Invalid words found: {', '.join(invalid_words)}. Answer must use only allowed nouns and verbs."
#     return "Answer follows lexical constraints."

# -----------------------
# AGENT SETUP
# -----------------------
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
TOOLS = [goalstep_tool_obj, spatial_tool_obj, sequence_generation_tool_obj, sequence_validation_tool_obj]

# This is being deprecated. 
#MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
MEMORY = MemorySaver()

# -----------------------
# RUN AGENT IN MAIN
# -----------------------
#Deprecated: use create_react_agent or agent_executor
# AGENT = initialize_agent(
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     tools= TOOLS,
#     llm=LLM_MODEL,
#     verbose=True,
#     memory=MEMORY,
#     handle_parsing_errors=True
# )
# Create the React agent with tools



def run_agent(target_activity, target_scene_graph, AGENT, AGENT_PROMPT):
    formatted_prompt = AGENT_PROMPT.format(target_activity=target_activity, target_scene_graph=target_scene_graph)
    return AGENT.run(formatted_prompt)

if __name__ == "__main__":
    # Fetch target space index and target_scene_graph
    target_video_idx = int(input("Input target index: "))
    target_spatial_video = spatial_test_video_list[target_video_idx]
    target_scene_graph = agent_input.extract_spatial_context(target_spatial_video)
    print(target_spatial_video["video_uid"])

    # Input target activity
    target_activity = "Cook soup"
    target_activity = input("Input target activity: ")

    # AGENT_PROMPT = ChatPromptTemplate.from_messages(
    #     "You are an action sequence planner agent that plans an action sequence comprising of multiple action steps for a user in his own environment. The user wants to perform a 'target_activity' as given below. The user is situated in a space where various entities are given by 'target_scene_graph' below.\n"
    #     "target_activity: {target_activity}"
    #     "target_scene_graph: {target_scene_graph}"
    #     "For generating action sequence to answer use the 'sequence_generation_tool_obj', base on target_activity and target_scene_graph. When examples are needed for more reasonable answers, there are two tools to look for:'goalsetp_tool_obj' and 'spatial_tool_obj'. Use the 'goalstep_tool_obj' to see which action steps can be taken for similar activities. Use the 'spatial_tool_obj' to see how states of entities can change for similarly set environments. When action sequence is generated, this sequence must be checked with the 'sequence_validation_tool_obj'. Unless 'sequence_validation_tool_obj' returns true, re-generate the answer with the 'sequence_generation_tool_obj'. Only finalize an answer if it passes the 'sequence_validation_tool_obj'"
    # )
    AGENT_PROMPT = ChatPromptTemplate.from_template(
        "You are an action sequence planner agent that plans an action sequence comprising of multiple action steps for a user in his own environment. The user wants to perform a 'target_activity' as given below. The user is situated in a space where various entities are given by 'target_scene_graph' below.\n"
        "target_activity: {target_activity}"
        "target_scene_graph: {target_scene_graph}"
        "For generating action sequence to answer use the 'sequence_generation_tool_obj', base on target_activity and target_scene_graph. When examples are needed for more reasonable answers, there are two tools to look for:'goalsetp_tool_obj' and 'spatial_tool_obj'. Use the 'goalstep_tool_obj' to see which action steps can be taken for similar activities. Use the 'spatial_tool_obj' to see how states of entities can change for similarly set environments. When action sequence is generated, this sequence must be checked with the 'sequence_validation_tool_obj'. Unless 'sequence_validation_tool_obj' returns true, re-generate the answer with the 'sequence_generation_tool_obj'. Only finalize an answer if it passes the 'sequence_validation_tool_obj'"
    )
    
    AGENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools."),
        ("user", "Query: {query}"),
        ("system", "Available tools: {tools}. Use them wisely."),
        ("system", "Tool names: {tool_names}"),
        ("assistant", "{agent_scratchpad}")  # Required for React agents
    ])

    AGENT = create_react_agent(
        tools=TOOLS,  # Register tools
        llm=LLM_MODEL,
        prompt=AGENT_PROMPT
        #agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use react-based agent
        #verbose=True,  # Enable verbose output for debugging
        #checkpointer=MEMORY,
        #handle parsing error not built in for this function.
    )

    # Run agent
    AGENT_EXECUTOR = AgentExecutor(agent=AGENT, tools=TOOLS, verbose=True)
    
    QUERY = ""
    response = AGENT_EXECUTOR.run(QUERY)
    print(run_agent(target_activity, target_scene_graph, AGENT, AGENT_PROMPT))

    # # -----------------------
    # # STREAMLIT UI
    # # -----------------------
    # def run_streamlit_app():
    #     """Run the Streamlit UI in the same Python script."""
    #     st.title("Agentic RAG with Streamlit")

    #     # Input box for user query
    #     query = st.text_area("Enter your query:", "")

    #     # Button to trigger the agent
    #     if st.button("Finalize Query"):
    #         if query:
    #             with st.spinner("Processing..."):
    #                 response = agent.run(query)
    #             st.subheader("Agent's Response:")
    #             st.write(response)
    #         else:
    #             st.error("Please enter a query to continue.")

    # if __name__ == "__main__":
        # run_streamlit_app()
