'''
func: predict deep activity for source scene, using source action sequece/scene graph/RAG examples
input: (source) action sequence, scene graph
output: source deep activity
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
#from langgraph.checkpoint.memory import MemorySaver
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import script_work.agent_database as agent_database
import script_work.agent_input as agent_input
import script_work.agent_query as agent_query
import script_work.agent_prompt as agent_prompt
from Scripts.Utils.util import util_constants
import workflow_data


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
# VIDEO LIST, VECSTORE, RETRIEVER
# -----------------------
# Load VIDEO LIST (use text video list for testing)
goalstep_test_video_list = workflow_data.goalstep_test_video_list
spatial_test_video_list = workflow_data.spatial_test_video_list

# LOAD FAISS VECSTORE
goalstep_vector_store = workflow_data.goalstep_vector_store
spatial_vector_store = workflow_data.spatial_vector_store

# MAKE base:VectorStoreRetriever
goalstep_retriever = goalstep_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
spatial_retriever = spatial_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


#------------------------
#Tools
#------------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    context = goalstep_retriever.invoke(query)
    return f"User Query: {query}. similar goalstep examples: {context}" 

def spatial_information_retriver(query:dict):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    context = spatial_retriever.invoke(query)
    return f"User Query: {query}. similar spatial examples: {context}"

# TODO source activity must be given as input
def move_down_activity(input: str):
    """Make deep activity more specific and concrete by lowering one level down its hierarchy"""
    input_dict = ast.literal_eval(input.strip())  # convert to python dict
    valid_json = json.dumps(input_dict, indent=4)  # read as JSON(wth "")
    input_json = json.loads(valid_json)
    query = input_json.get("query")
    source_action_sequence = input_json.get("source_action_sequence")
    source_scene_graph = input_json.get("source_scene_graph")
    source_activity = input_json.get("source_activity")

    prompt = f"Here is the query: {query}. Here is the source_action_sequence: {source_action_sequence}. Here is the source_scene_graph: {source_scene_graph}. Here is the source_activity: {source_activity}"

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system", 
            "content": "Return a serial of noun or words which is a specific and deep hierarchical description of source activity. For example, consider that ""cook steak"" is input activity, and sequence is about cooking steak. Ask yourself what steak? Going down with steak can give you more information on animal types. Then you can go deeper if there is information on areas or specific cut of meat(e.g. tomahawk, sirloin, etc). This can result in different types of cuisines finally. The final example can end like ""(cook)(steak)(pork)(loin)(cutlet)"". You cook steak which is pork, which is loin, which is cutlet. Final answer must be given as multiple small brackets enclosing a word. (verb)(noun) is a format used to describe input source activity. (verb)(noun)(noun)....(noun) format is used for final answer called categorized_activity. When you make your answer, start from (verb)(noun) for the input activity, and then order the remaining (noun) from the highest category to the narrowest one!. For our example just return ""(cook)(steak)(pork)(loin)(cutlet)"" and nothing else! Like this!  For this example 'output': ""(cook)(steak)(pork)(loin)(cutlet)"""
            }, 
            { "role": "user", "content": prompt}
                ],
        temperature=0.5
    )

    categorized_activity = response.choices[0].message.content.strip()

    # return f"Thought: Here is the categorized activity.\nAction: move_down_activity_tool\nAction Input: {json.dumps({'query': query, 'source_action_sequence': source_action_sequence, 'source_scene_graph': source_scene_graph})}\n{categorized_activity}"
    return f"Thought: Here is the categorized activity.\n{categorized_activity}"


# -----------------------
# AGENT SETUP
# -----------------------
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
#LLM_MODEL_OLLAMA = OllamaLLM()
TOOLS = [
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
        name = "move_down_activity_tool",
        func = move_down_activity,
        description = "This tool generates a spefic and deep hierarchical description of source activity"
    ),
    ]


def run_agent(source_video_idx=None, source_activity=""):
    """
    set input arguments
    set agent arguments
    create react agent
    create agent executor
    pass on responese
    """
    if source_video_idx is None:
        source_video_idx = int(input("Input source index:"))
    if source_activity is "":
        source_activity = input("Input source activity")
    source_goalstep_video = goalstep_test_video_list[source_video_idx]
    source_spatial_video = spatial_test_video_list[source_video_idx]
    source_action_sequence = agent_input.extract_lower_goalstep_segments(source_goalstep_video)
    source_scene_graph = agent_input.extract_spatial_context(source_spatial_video)
    tool_names =", ".join([t.name for t in TOOLS])    
    
    QUERY = "Categorically describe source activity in a very specific way."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=LLM_MODEL,
        prompt=agent_prompt.AGENT1b_PROMPT
    )    

    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS, 
        verbose=True, 
        handle_parsing_errors=True,
        memory=MEMORY
    )

    response = AGENT_EXECUTOR.invoke(
        {
            "query": QUERY, 
            "source_scene_graph": source_scene_graph,
            "source_action_sequence": source_action_sequence,
            "tools": TOOLS,  # Pass tool objects
            "tool_names": tool_names,  # Convert list to comma-separated string
            "agent_scratchpad": ""  # Let LangChain handle this dynamically
         },
        config={"max_iterations": 5}
    )
    return response


if __name__ == "__main__":

    source_video_idx = int(input("Input source index:"))
    response = run_agent(source_video_idx)

    if "move_down_activity_tool" in response:
        response =  response.split("Action Input:")[1].strip()

    print(f"response {response}")