'''
This is an Agentic RAG for recognition of activity based on input action sequence and spatial context
input: source action sequence, source scene graph
context: additional vector dataset
output: source activity
'''
import sys
import os
import openai
import logging
import json
import ast
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
from langgraph.checkpoint.memory import MemorySaver
#packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
sys.path.append(os.path.abspath('/usr/local/lib/python3.10/dist-packages'))
import script_work.agent_database as agent_database
import script_work.agent_input as agent_input
import script_work.agent_query as agent_query
import script_work.agent_prompt as agent_prompt
from util import util_constants
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
# CONFIGURE DATA
# -----------------------
ALLOWED_WORDS = {
    "nouns": {"apple", "banana", "computer", "data", "research"},
    "verbs": {"calculate", "analyze", "study", "process"}
}

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

def activity_prediction(input: str):
    """Predict an activity of the user based on the input"""
    input_dict = ast.literal_eval(input.strip())  # convert to python dict
    valid_json = json.dumps(input_dict, indent=4)  # read as JSON(wth "")
    input_json = json.loads(valid_json)
    query = input_json.get("query")
    source_action_sequence = input_json.get("source_action_sequence")
    source_scene_graph = input_json.get("source_scene_graph")

    prompt = f"Here is the query: {query}. Here is the source_action_sequence: {source_action_sequence}. Here is the source_scene_graph: {source_scene_graph}"

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system", 
            "content": "You predict current user activity based on input with five items. The activity is a ONE PHRASE SUMMARY of the source_action_sequence in the input query. First input is query. Second input argument is source_action_sequence, given in the prompt. Third input argument is source_scene_graph, also given in system prompt. Fourth input argument is relevant_goalstep_information, sometimes given as None. Fifth input argument is relevant_scene_graph, sometimes given as None. Activity MUST be given in one phrase inside a double quote. Only one verb is allowed."
            }, 
            { "role": "user", "content": prompt}
                ],
        temperature=0.5
    )

    activity = response.choices[0].message.content.strip()

    return f"Thought: The activity is predicted.\nAction: activity_prediction_tool\nAction Input: {json.dumps({'query': query, 'source_action_sequence': source_action_sequence, 'source_scene_graph': source_scene_graph})}\n{activity}"


# -----------------------
# DEFINE TOOLS with TOOLFUNC
# -----------------------
goalstep_retriever_tool = Tool(
    name = "goalstep_retriever_tool",
    func = goalstep_information_retriever,
    description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
)

spatial_retriever_tool = Tool(
    name = "spatial_retriever_tool",
    func = spatial_information_retriver,
    description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
)

activity_prediction_tool = Tool(
    name = "activity_prediction_tool",
    func = activity_prediction,
    description = "Activity prediction tool, which can summarize the sequential multiple actions into a short single phrase of activity. . Additional examples from other environments can also be used. Input: query(str), target_activity(str), . Output: action_sequence_steps(str)"
)


# -----------------------
# AGENT SETUP
# -----------------------
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1)
LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
#LLM_MODEL_OLLAMA = OllamaLLM()
TOOLS = [
    goalstep_retriever_tool,
    spatial_retriever_tool,
    activity_prediction_tool,
    ]
MEMORY = MemorySaver()


if __name__ == "__main__":
    # -----------------------
    # AGENT INPUT ARGUMENTS
    # -----------------------
    source_video_idx = int(input("Input source index:"))
    source_goalstep_video = goalstep_test_video_list[source_video_idx]
    source_spatial_video = spatial_test_video_list[source_video_idx]
    source_action_sequence = agent_input.extract_lower_goalstep_segments(source_goalstep_video)
    source_scene_graph = agent_input.extract_spatial_context(source_spatial_video)
    
    tool_names =", ".join([t.name for t in TOOLS])





    # -----------------------
    # AGENT PROMPT
    # -----------------------
    AGENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an agent that answers queries using tools. If you have gathered enough information, respond with:
        
        Thought: I now have enough information to answer.
        Final Answer: [Your answer]
         
        Otherwise, use this format:
         Thought: [Your reasoning]
         Action: [Tool name]
         Action Input: {{"query": "{query}", "source_action_sequence": "{source_action_sequence}", "source_scene_graph": "{source_scene_graph}" }}
         """),

        ("system", "The user is performing a sequence of actions in this form: {source_action_sequence}."),
        ("system", "The user is in a space described by this scene graph. Predicted activity must be able to be performed in this scene: {source_scene_graph}."),
        ("system", "Available tools: {tools}. Use them wisely. Actively use retrieval tools to come up with plausible answer."),
        ("system", "Tool names: {tool_names}"),  # Required for React agents
        ("user", "{query}"),  # The user query should be directly included
        ("assistant", "{agent_scratchpad}")  # Required for React agents
    ])

    #Example in Agent Prompt is to be deprecated
        #      Example:
        #  Thought: I need to predict activity from the input.
        #  Action: activity_prediction_tool
        #  Action Input: {{"query": "Predict activity from the action sequence and the scene graph", "source_action_sequence": "{{source_action_sequence}}", "source_scene_graph": "{{source_scene_graph}}"}}
        #  """),

    # -----------------------
    # CREATE & RUN AGENT IN MAIN
    # -----------------------
    AGENT = create_react_agent(
        tools=TOOLS,  # Register tools
        llm=LLM_MODEL,
        prompt=AGENT_PROMPT
        #agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use react-based agent
        #verbose=True,  # Enable verbose output for debugging
        #checkpointer=MEMORY,
        #handle parsing error not built in for this function.
    )

    AGENT_EXECUTOR = AgentExecutor(
        agent=AGENT, 
        tools=TOOLS, 
        verbose=True, 
        handle_parsing_errors=True
    )

    QUERY = "Give me a sequence of actions to fulfill the target_activity inside the environment of target_scene_graph"

    response = AGENT_EXECUTOR.invoke({
        "query": QUERY,
        "source_action_sequence": source_action_sequence,
        "source_scene_graph": source_scene_graph,
        "tools": TOOLS,  # Pass tool objects
        "tool_names": ", ".join(tool_names),  # Convert list to comma-separated string
        "agent_scratchpad": ""  # Let LangChain handle this dynamically
    })

    print(f"response {response}")





# # -----------------------
# # AGENT SETUP
# # -----------------------
# # LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4o-mini", temperature=1) #10x cheaper
# LLM_MODEL = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4", temperature=1)
# TOOLS = [retrieve_relevant_docs_goalstep, retrieve_relevant_docs_spatial]
# MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# AGENT_PROMPT = ChatPromptTemplate.from_messages(
#     "You are an AI assistant agent that uses multiple tools to answer user query.\n"
#     "Use the 'retrieve_relevant_docs' tool to fetch information before answering.\n"
#     "Whenever you generate an answer, use the following tools to verify it:\n"
#     "- 'check_answer_relevance' to ensure the answer actually answers the question.\n"
#     "- 'enforce_lexical_constraints' to verify that the answer only contains allowed words.\n"
#     "Only finalize an answer if it passes both checks.\n"
#     "User Query: {query}"
# )


# # -----------------------
# # RUN AGENT IN MAIN
# # -----------------------
# AGENT = initialize_agent(
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     tools= TOOLS,
#     llm=LLM_MODEL,
#     verbose=True,
#     memory=MEMORY,
#     handle_parsing_errors=True
# )

# def run_agent(input_text):
#     formatted_prompt = AGENT_PROMPT.format(query=input_text)
#     return AGENT.run(formatted_prompt)

# if __name__ == "__main__":
#     agent1_query = "What if the activity of the "
#     print(run_agent("Start process"))