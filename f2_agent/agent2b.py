'''
func: specify deep activity to target level
input: common deep activity
output: target deep activity
'''
import sys
import os
import openai
import logging
import json
import ast
import argparse
from dotenv import load_dotenv
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
from langchain.agents import AgentType, create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
#from langgraph.checkpoint.memory import MemorySaver # Saves everyghing leading to overflow
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import f1_init.agent_init as agent_init
import f1_init.agent_input as agent_input
import f2_agent.agent_prompt as agent_prompt
import workflow_data



# -----------------------
# Path & API & Model
# -----------------------
logging.basicConfig(level=logging.ERROR)
load_dotenv()
parser_stroutput = StrOutputParser()

openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL = agent_init.LLM_MODEL_4MINI
LLM_MODEL_AGENT = agent_init.LLM_MODEL_4MINI


# -----------------------
# VIDEO LIST, VECSTORE, RETRIEVER
# -----------------------
# Load VIDEO LIST
goalstep_test_video_list = workflow_data.goalstep_test_video_list
spatial_test_video_list = workflow_data.spatial_test_video_list

# LOAD FAISS VECSTORE
goalstep_vector_store = workflow_data.goalstep_vector_store
spatial_vector_store = workflow_data.spatial_vector_store

# MAKE base:VectorStoreRetriever
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# -----------------------
# TOOL FUNCTION
# -----------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    context = goalstep_retriever.invoke(query)
    return f"User Query: {query}. similar goalstep examples: {context}" 

def spatial_information_retriver(query:dict):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    context = spatial_retriever.invoke(query)
    return f"User Query: {query}. similar spatial examples: {context}"

def check_executability(query:str):
    """Test if goal of deep activity can be met in current target_scene."""
    #LLM LOGIC TO SEE THIS

def move_down_activity(query:str):
    """Make deep activity more specific and concrete by lowering one level down its hierarchy"""
    #LLM LOGIC

# -----------------------
# AGENT SETUP
# -----------------------
MEMORY = ConversationBufferWindowMemory(k=3)
#MEMORY = MemorySaver() # Saves everyghing leading to overflow
TOOLS_2b = [
    Tool(
        name = "goalstep_retriever_tool",
        func = goalstep_information_retriever,
        description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
    ),
    Tool(
        name = "spatial_retriever_tool",
        func = spatial_information_retriver,
        description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
    ),
    Tool(
        name = "executability_check_tool",
        func = check_executability,
        description = "Test if goal of deep activity can be met in current target_scene."
    ),
    Tool(
        name = "move_down_activity_tool",
        func = move_down_activity,
        description = "Make deep activity more specific and concrete by lowering one level down its hierarchy"
    )
    ]




if __name__ == "__main__":
    # -----------------------
    # ARGS / CREATE / EXECUTE / RES
    # -----------------------
    source_video_idx = int(input("Input source index:"))
    source_goalstep_video = goalstep_test_video_list[source_video_idx]
    source_spatial_video = spatial_test_video_list[source_video_idx]
    source_action_sequence = agent_input.extract_lower_goalstep_segments(source_goalstep_video)
    source_scene_graph = agent_input.extract_spatial_context(source_spatial_video)
    tool_names =", ".join([t.name for t in TOOLS_2b])

    AGENT = create_react_agent(
        tools=TOOLS_2b,
        llm=LLM_MODEL,
        prompt=agent_prompt.AGENT1_PROMPT
    )

    AGENT_EXECUTOR = AgentExecutor(
        agent=LLM_MODEL_AGENT, 
        tools=TOOLS_2b, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=MEMORY
    )

    QUERY = "Give me a sequence of actions to fulfill the target_activity inside the environment of target_scene_graph"
    response = AGENT_EXECUTOR.invoke(        
        {
            "query": QUERY,
            "source_action_sequence": source_action_sequence,
            "source_scene_graph": source_scene_graph,
            "tools": TOOLS_2b,  # Pass tool objects
            "tool_names": ", ".join(tool_names),  # Convert list to comma-separated string
            "agent_scratchpad": ""  # Let LangChain handle this dynamically
        },
            max_iterations = 3
    )

    print(f"response {response}")




#openai client
    # client = openai.OpenAI()
    # response = client.chat.completions.create(
    #     model=LLM_MODEL_AGENT
    #     messages=[
    #         {
    #         "role": "system", 
    #         "content": "Return a serial of noun or words which is a specific and deep hierarchical description of source activity. For example, consider that ""cook steak"" is input activity, and sequence is about cooking steak. Ask yourself what steak? Going down with steak can give you more information on animal types. Then you can go deeper if there is information on areas or specific cut of meat(e.g. tomahawk, sirloin, etc). This can result in different types of cuisines finally. The final example can end like ""(cook)(steak)(pork)(loin)(cutlet)"". You cook steak which is pork, which is loin, which is cutlet. Final answer must be given as multiple small brackets enclosing a word. (verb)(noun) is a format used to describe input source activity. (verb)(noun)(noun)....(noun) format is used for final answer called categorized_activity. When you make your answer, start from (verb)(noun) for the input activity, and then order the remaining (noun) from the highest category to the narrowest one!. For our example just return ""(cook)(steak)(pork)(loin)(cutlet)"" and nothing else! Like this!  For this example 'output': ""(cook)(steak)(pork)(loin)(cutlet)"""
    #         }, 
    #         { "role": "user", "content": prompt}
    #             ],
    #     temperature=0.5
    # )
    # categorized_activity = response.choices[0].message.content.strip()
    # return f"Thought: Here is the categorized activity.\n{categorized_activity}"    


    #Ollama Response 
    #     response = ollama.chat(
    #     model = LLM_MODEL_AGENT,
    #     messages=[
    #         {
    #         "role": "system", 
    #         "content": "You predict current user activity based on five input items. Activity MUST be given in one phrase inside a double quote. Answer format is as follows {{action in form of verb}} {{target in form of noun}}"
    #         }
    #     ],
    #     options={
    #         'temperature':0.5
    #     }
    # )
    # activity = response['message']['content']