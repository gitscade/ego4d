import sys
import os
import openai
import logging
import json
import ast
import argparse
from dotenv import load_dotenv
import pickle
#llm
import ollama
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
from langchain.prompts import ChatPromptTemplate
import f1_init.agent_init as agent_init
import f1_init.database_init as database_init
import f2_agent.agent_prompt as agent_prompt
import util.util_funcs as util_funcs
import f1_init.constants_init as constants_init
import re

#------------------------
#prompt messages
#------------------------
def get_agent1a_message(inputs:list):
    """
    func: returns prompt and messages used for agent1a\n
    input: single list: [tools, sequence, scenegraph]\n
    return: AGENT1a_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    query = "summarize the input action sequence with a single verb and a single noun"
    tool_names =", ".join([t.name for t in tools])    

    AGENT1a_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful taxonomy summarizer that summarizes an action_sequence in input scene_graph, using tools. State your final answer as a string in a section labeled 'Final Answer:'.:
        
            "Final Answer: [Your answer]"
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: [The input to the action]

        """),

        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "{query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )

    MESSAGE_ACTIVITY_PREDICTION = [
            {"role": "system", "content": "You are a linguist that summarizes current user activity to a single verb and a single noun"}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"}        
        ]
    return AGENT1a_PROMPT, MESSAGE_ACTIVITY_PREDICTION

def get_agent3_message(inputs:list):
    """
    get agent3 inputs
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]
    source_core_activity = inputs[4]  
    query = "predict target_action_sequence that can realize the target_activity_taxonomy in the target_scene_graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT3_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful action planner that constructs an action sequence for the target_scene_graph to achieve the target_activity_taxonomy, using tools. Return the final action_sequence as a target_action_sequence, following the format below. USE SAME FORMAT AS THE source_action_sequence WITHOUT ADDING ANYTHING MORE:
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}]
        """),
        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the source scene graph: {source_scene_graph}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),   
        ("system", "This is the source core activity: {source_core_activity}"),        
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )

    MESSAGE_TARGET_SEQUENCE_PREDICTION = [
            {"role": "system", "content": """You are a action planner expert that makes action sequence in a target_scene_graph to achieve the context of the core_activity. Let me give you a step-by-step example of how you function.
             
             First, you receive a source_core_activity as follows. This is a final goal of the target_action_sequence you are making:

             "Cook Steak"

             You also receive a source_action_sequence like this, which describes a sequence of actions inside a source_scene_graph.

            [
             "put pan on stove",
             "add oil on pan",
             "turn on the stove",
             "put meat on stove",
             "salt the meat"
             ]

             You will also receive source_scene_graph, in which the above source_action_sequence and source_core_acvitity is performed.
             
             You will then receive target_scene_graph, for which you will have to make an action sequence for.
            
             You have to make an action sequence that follows context of the source_core_activity, while trying to be similar to the source_action sequence as much as possible. However, you can only use entities in the target_scene_graph. 
             
             You will generate the final target_action_sequence similar to this format:

             ["action1", "action2", ..., "actionN"]
             
             STRICTLY Follow the format below to print the output:
             
             Final Answer: ["action1", "action2", ..., "final action]
             """}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
            {"role": "user", "content": f"Here is the source core activity:\n{source_core_activity}\n"}          
        ]   
    return AGENT3_PROMPT, MESSAGE_TARGET_SEQUENCE_PREDICTION
#------------------------
#Tool Funcs
#------------------------
def goalstep_information_retriever(source_action_sequence:str):
    """Retrieve the most relevant goalstep dataset documents based on input of source_action_sequence."""
    context = agent_init.goalstep_retriever.invoke(source_action_sequence)
    return f"source_action_sequence: {source_action_sequence}. similar goalstep examples: {context}" 

def spatial_information_retriver(source_scene_graph:dict):
    """Retrieve the most relevant spatial context documents based on input of source_scene_graph"""
    context = agent_init.spatial_retriever.invoke(source_scene_graph)
    return f"source_scene_graph: {source_scene_graph}. similar spatial examples: {context}"

def activity_prediction(MESSAGE_ACTIVITY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR):
    """Predict an core summary of the user activity based on input"""
    my_temperature = 0.5
    try:
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_ACTIVITY_PREDICTION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_ACTIVITY_PREDICTION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']        

    except Exception as e:
        return f"Error: activity_prediction: {str(e)}"

def predict_target_action_sequence(MESSAGE_TARGET_SEQUENCE_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR):
    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_TARGET_SEQUENCE_PREDICTION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_TARGET_SEQUENCE_PREDICTION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: action_sequence_prediction: {str(e)}"


# -----------------------
# Tool GET Funcs
# -----------------------
def get_agent1a_tools():
    """
    return tools
    """
    tools=[
    # Tool(
    #     name = "goalstep_retriever_tool",
    #     func = goalstep_information_retriever,
    #     description = "Retrieves relevant activity step information in other scenes."
    # ),
    # Tool(
    #     name = "spatial_retriever_tool",
    #     func = spatial_information_retriver,
    #     description = "Retrieves relevant scene and entity information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
    # ),
    Tool(
        name = "activity_prediction_tool",
        func = lambda _: activity_prediction(MESSAGE_ACTIVITY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR),
        description = "Activity prediction tool, which can summarize the sequential multiple actions into a single verb and a single noun."
    ),
    # Tool(
    #     name = "reorder_activity_taxonomy_tool",
    #     func = lambda _: reorder_activity_taxonomy(MESSAGE_REORDER_TAXONOMY),
    #     description = "reorder_activity_taxonomy, which can reorder the input taxonomy."
    # ),    
    ]
    return tools

def get_agent3_tools():
    """
    return tools
    """
    tools = [
        # Tool(
        #     name = "goalstep_retriever_tool",
        #     func = goalstep_information_retriever,
        #     description = "Retrieves relevant goalstep information in other environments, where similar activities are performed in steps. "
        # ),
        # Tool(
        #     name = "spatial_retriever_tool",
        #     func = spatial_information_retriver,
        #     description = "Retrieves relevant spatial information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
        # ),
        Tool(
            name = "predict target action sequence",
            func = lambda _: predict_target_action_sequence(MESSAGE_TARGET_SEQUENCE_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR),
            description = "make target action sequence that performs the core activity and target_activity_taxonomy in target_scene."
        ),
        ]    
    return tools

# -----------------------
# Agent Function
# -----------------------
def run_agent_1a(input, agent_llm_chat):
    """"
    func: run agent 1a with source video idx info\n
    input: [tools_1a, AGENT1a_PROMPT, source_action_sequence, source_scene_graph]\n
    output: response
    """
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    TOOLNAMES =", ".join([t.name for t in TOOLS])    

    # AGENT
    QUERY = "Give me an phrase describing the activity of source_action_sequence."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!
    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_llm_chat,
        prompt=AGENT_PROMPT
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
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": ""
         },
        config={"max_iterations": 5}
    )
    return response

def run_agent3(input, agent_llm_chat):
    """"
    func: run agent 3\n
    input: [tools_3, AGENT3_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, target_activity_taxonomy, source_core_taxonomy]\n
    output: target_action_sequence
    """
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    target_scene_graph = input[4]
    source_core_activity = input[5]

    TOOLNAMES =", ".join([t.name for t in TOOLS])
    QUERY = "Predict Action Sequence for the target scene graph"    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_llm_chat,
        prompt=AGENT_PROMPT
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
            "source_action_sequence": source_action_sequence,
            "source_scene_graph": source_scene_graph,   
            "target_scene_graph": target_scene_graph,            
            "source_core_activity": source_core_activity,           
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

def run_agent4(source_core_activity, target_action_sequence, TOOL_LLM_API, TOOL_LLM_STR):    

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": """You are a helpful assistant for testing if source_core_action is summarized representation of target_activity_taxonomy and target_action_sequence. 
            
            source_core_action variable comprises of a single noun and a verb.

            Your task is
            1. Examine if target_action_sequence is well represented by the source_core_action, so that source_core_activity can be seen as a goal summarizing the target_action sequence.

            Answer to your question in "yes" and "no" in this format below.:

            taxonomy_representation: [yes or no]
            sequence_representation : [yes or no]
            
            If not, return '''no'''"""},
        {"role": "user", "content":  "Here is the source_core_activity: \n{source_core_activity}\n"},
        {"role": "user", "content":  "Here is the target_action_sequence: \n{target_action_sequence}\n"},
    ]
    )

    # Format the prompt
    ("AGENT4: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages(
        source_core_activity=source_core_activity,
        target_action_sequence=target_action_sequence
    )

    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            llm = ChatOpenAI(model=TOOL_LLM_STR, temperature=my_temperature)             
            response = llm.invoke(formatted_messages)
            # print(f"my response : {response.content}")   
            return response.content

        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= prompt,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: action_sequence_prediction: {str(e)}"


if __name__ == "__main__":
    # -----------------------
    # TEST SETTINGS
    # -----------------------
    agent_api_name = "openai"
    agent_model_name = "gpt-4.1"
    tool_api_name = "openai"
    tool_model_name = "gpt-4.1"    
    AGENT_LLM_API, AGENT_LLM_STR, AGENT_LLM_CHAT = agent_init.SET_LLMS(agent_api_name, agent_model_name, temperature=0.2)
    TOOL_LLM_API, TOOL_LLM_STR, TOOL_LLM_CHAT = agent_init.SET_LLMS(tool_api_name, tool_model_name, temperature=0.2)

    # # SETUP FIRST INPUTS
    BASELINE_FOLDER = "/output-1goalmediation/"
    BASELINE_FOLDER = "/output-1goalmediation-norag-0602/"
    PATH_SOURCE_TARGET_OUTPUT = constants_init.PATH_SOURCE_TARGET + BASELINE_FOLDER

    # Get scenegraph for source and target
    source_spatial_json_list, target_spatial_json_list, aug_levels = agent_init.get_paired_spatial_json_list(constants_init.PATH_AUGMENTATION_v6)

    # Make source_idx_list that matches length of the above json list
    source_idx_list = [i for i in range(len(source_spatial_json_list)//len(aug_levels)) for _ in range(len(aug_levels))]

    # # for i in range(0, len(source_list)):
    for i in range(len(source_idx_list)):

        # -----------------------
        # CHECK PATHS
        # -----------------------
        PATH_SOURCEINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_sourceinfo.pkl"
        PATH_TARGETINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_targetinfo.pkl"
        PATH_AGENT1a = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent1a.pkl"
        PATH_AGENT3 = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent3.pkl"
        PATH_AGENT4 = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent4.pkl"  

        # init path bools
        bool_sourceinfo = False
        bool_targetinfo = False
        bool_agent1a = False
        bool_agent3 = False
        bool_agent4 = False

        # check file with paths
        bool_sourceinfo = agent_init.check_file(PATH_SOURCEINFO)
        bool_targetinfo = agent_init.check_file(PATH_TARGETINFO)
        bool_agent1a = agent_init.check_file(PATH_AGENT1a)
        bool_agent3 = agent_init.check_file(PATH_AGENT3)
        bool_agent4 = agent_init.check_file(PATH_AGENT4)

        # if every file exist, break from this whole loop
        if bool_sourceinfo and bool_targetinfo and bool_agent1a and  bool_agent3 and bool_agent4:
            continue   
        else:
            print(f"{i} missing")

        # if no file whatsoever, bool_runall is True to run everything without loading
        if not bool_sourceinfo and not bool_targetinfo and not bool_agent1a and not bool_agent3 and not bool_agent4:
            print(f"{i} running every agent")
            bool_runall = True

        # prepare necessary files
        source_video_idx = source_idx_list[i]
        source_action_sequence, scenegraphnotused = agent_init.get_video_info(source_video_idx)
        source_scene_graph = agent_init.extract_spatial_context(source_spatial_json_list[i])
        target_scene_graph = agent_init.extract_spatial_context(target_spatial_json_list[i])
        source_uid  = source_spatial_json_list[i]['video_id']
        target_uid = target_spatial_json_list[i]['video_id']
        spatial_similarity  = target_spatial_json_list[i]['spatial_similarity']

        # if os.path.exists(PATH_SOURCEINFO):
        #     os.remove(PATH_SOURCEINFO)
        # with open(PATH_SOURCEINFO, 'wb') as f:
        #     print(f"{i} ")
        #     dict = {"source_idx": source_video_idx, "source_uid": source_uid, "source_action_sequence": source_action_sequence, "source_scene_graph": source_scene_graph, "spatial_similarity": spatial_similarity}
        #     pickle.dump(dict, f)

        # if os.path.exists(PATH_TARGETINFO):
        #     os.remove(PATH_TARGETINFO)            
        # with open(PATH_TARGETINFO, 'wb') as f:
        #     print(f"{i} ")
        #     dict = {"target_idx": (source_video_idx+10)%71, "target_uid": target_uid, "target_scene_graph": target_scene_graph}
        #     pickle.dump(dict, f)


        # sourceinfo and targetinfo
        while not bool_sourceinfo and not bool_targetinfo:
            with open(PATH_SOURCEINFO, 'wb') as f:
                dict = {"source_idx": source_video_idx, "source_uid": source_uid, "source_action_sequence": source_action_sequence, "source_scene_graph": source_scene_graph, "spatial_similarity": spatial_similarity}
                pickle.dump(dict, f)       
                bool_sourceinfo = True

            with open(PATH_TARGETINFO, 'wb') as f:
                dict = {"target_idx": (source_video_idx+10)%71, "target_uid": target_uid, "target_scene_graph": target_scene_graph}
                pickle.dump(dict, f)
                bool_targetinfo = True


        # -----------------------
        # AGENT1a: PREDICT CORE ACTIVITY
        # -----------------------
        if bool_agent1a:
            with open(PATH_AGENT1a, 'rb') as f:
                source_core_activity = pickle.load(f)
        else:
            while not bool_agent1a:
                try:
                    tools_1a = get_agent1a_tools()

                    input_1a_message = [tools_1a, source_action_sequence, source_scene_graph]
                    AGENT1a_PROMPT, MESSAGE_ACTIVITY_PREDICTION = get_agent1a_message(input_1a_message)
                    input_1a_agent = [tools_1a, AGENT1a_PROMPT, source_action_sequence, source_scene_graph]
                    response_1a = run_agent_1a(input_1a_agent, AGENT_LLM_CHAT)
                    source_core_activity = response_1a['output']

                    with open(PATH_AGENT1a, 'wb') as f:
                        pickle.dump(source_core_activity, f)        
                        print(f"sgent1a saved: source_core_activity")
                        bool_agent1a = True

                except Exception as e:
                    print(f"Agent1a failed at index {i}: {e}")
                    continue
        
        # -----------------------
        # AGENT3: PREDICT TARGET ACTION SEQUENCE
        # -----------------------
        print(f"AGENT3")        
        if bool_agent3:
            with open(PATH_AGENT3, 'wb') as f:
                target_action_sequence = pickle.load(f)
        else:
            while not bool_agent3:
                try:         
                    # TARGET_SCENE_EXAMPLE->RAG: SPATIAL EXAMPLE FOR ONLY TARGET_SCENE             
                    tools_3 = get_agent3_tools()
                    input3_message = [tools_3, source_action_sequence, source_scene_graph, target_scene_graph, source_core_activity]
                    AGENT3_PROMPT, MESSAGE_TARGET_SEQUENCE_PREDICTION=get_agent3_message(input3_message)       
                    input3_agent = [tools_3, AGENT3_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_core_activity]   
                    response_3 = run_agent3(input3_agent, AGENT_LLM_CHAT)
                    target_action_sequence = response_3['output']

                    # SERIALIZE TO FORMAT
                    print(f"3b output {target_action_sequence}")
                    target_action_sequence = re.sub(r"^```json\s*|\s*```$", "", target_action_sequence.strip())
                    target_action_sequence = util_funcs.jsondump_agent_response(target_action_sequence)

                    with open(PATH_AGENT3, 'wb') as f:
                        pickle.dump(target_action_sequence, f)        
                        print(f"agent3 saved: target_action_sequence")
                        bool_agent3 = True

                except Exception as e:
                    print(f"Agent3 failed at index {i}: {e}")
                    continue


        # -----------------------
        # FINAL: TEST FAITHFULNESS TO CORE ACTIVITY
        # -----------------------
        if bool_agent4:
            with open(PATH_AGENT4, 'wb') as f:
                final_response = pickle.load(f)
        else:
            while not bool_agent4:
                try:
                    final_response = run_agent4(
                        source_core_activity,
                        target_action_sequence,
                        agent_api_name,
                        agent_model_name
                    )

                    print(final_response)
                    with open(PATH_AGENT4, 'wb') as f:
                        pickle.dump(final_response, f)        
                        print(f"agent4 saved: final answer {final_response}")
                        bool_agent4 = True       

                except Exception as e:
                    print(f"Agent4 failed at index {i}: {e}")
                    continue                    