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
import agents_llmfuncs as llmfuncs
import re

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

# -----------------------
# Agent Function
# -----------------------
def run_core_action_test(source_core_activity, target_activity_taxonomy, target_action_sequence, TOOL_LLM_API, TOOL_LLM_STR):    

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": """You are a helpful assistant for testing if source_core_action is summarized representation of target_activity_taxonomy and target_action_sequence. 
            
            source_core_action variable comprises of a single noun and a verb.

            target_activity_taxonomy is a description of the single noun, with 5-level taxonomy of key-value pairs, where key is a classification type, and value is the details on the classification.

            target_action_sequence, is a sequence of actions.

            Your task is twofold
            1. Examine if target_activity_taxonomy is a faithful description of the noun in the source_core_activity.
            2. Examine if target_action_sequence is well represented by the source_core_action, so that source_core_activity can be seen as a goal summarizing the target_action sequence.

            Answer to your question in "yes" and "no" in this format below.:

            taxonomy_representation: [yes or no]
            sequence_representation : [yes or no]
            
            If not, return '''no'''"""},
        {"role": "user", "content":  "Here is the source_core_activity: \n{source_core_activity}\n"},
        {"role": "user", "content":  "Here is the target_activity_taxonomy: \n{target_activity_taxonomy}\n"},
        {"role": "user", "content":  "Here is the target_action_sequence: \n{target_action_sequence}\n"},
    ]
    )

    # Format the prompt
    ("AGENT4: FORMATTING MESSAGES")
    formatted_messages = prompt.format_messages(
        source_core_activity=source_core_activity,
        target_activity_taxonomy=target_activity_taxonomy,
        target_action_sequence=target_action_sequence
    )

    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            llm = ChatOpenAI(model=TOOL_LLM_STR, temperature=my_temperature)             
            response = llm.invoke(formatted_messages)
            print(f"my response : {response.content}")   
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

    # PATHS
    trial_idx_input = input("Trial Index(0 to 9)")
    BASELINE_FOLDER = "/output0" + f"_{trial_idx_input}" +  "/"
    PATH_SOURCE_TARGET_OUTPUT = constants_init.PATH_SOURCE_TARGET + BASELINE_FOLDER
    if not os.path.exists(PATH_SOURCE_TARGET_OUTPUT):
        print(f"making {PATH_SOURCE_TARGET_OUTPUT}")
        os.makedirs(PATH_SOURCE_TARGET_OUTPUT)  

    target_folders = [
        constants_init.PATH_AUGMENTATION_v8_0th,
        constants_init.PATH_AUGMENTATION_v8_1th,
        constants_init.PATH_AUGMENTATION_v8_2th,
        constants_init.PATH_AUGMENTATION_v8_3th,
        constants_init.PATH_AUGMENTATION_v8_4th,
        constants_init.PATH_AUGMENTATION_v8_5th,
        constants_init.PATH_AUGMENTATION_v8_6th,
        constants_init.PATH_AUGMENTATION_v8_7th,
        constants_init.PATH_AUGMENTATION_v8_8th,
        constants_init.PATH_AUGMENTATION_v8_9th,
    ]
    source_folder = constants_init.PATH_AUGMENTATION_v8_source
    target_folder = target_folders[int(trial_idx_input)]
    source_spatial_json_list, target_spatial_json_list = agent_init.get_paired_spatial_json_list_v8(source_folder, target_folder)

    # 
    aug_levels = ['0','0.2','0.4','0.6','0.8','1.0']
    trial_index_levels = ['th']
    

    # # Make source_idx_list that matches length of the above json list
    source_idx_list = [i for i in range(len(source_spatial_json_list)//(len(aug_levels)*len(trial_index_levels))) for _ in range(len(aug_levels))]
    print(len(source_spatial_json_list))


    # # for i in range(0, len(source_list)):
    for i in range(len(source_spatial_json_list)):
    # for i in range(10):

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
        # bool_sourceinfo = agent_init.check_file(PATH_SOURCEINFO)
        # bool_targetinfo = agent_init.check_file(PATH_TARGETINFO)
        bool_agent1a = agent_init.check_file(PATH_AGENT1a)
        bool_agent3 = agent_init.check_file(PATH_AGENT3)
        bool_agent4 = agent_init.check_file(PATH_AGENT4)

        # If all files exist move to next idx
        if bool_sourceinfo and bool_targetinfo and bool_agent1a and bool_agent3:
            print(f"{i} continue")
            continue        
        else:
            print(f"{i} missing")
            print(f"{bool_sourceinfo} {bool_targetinfo} {bool_agent1a} {bool_agent3}")

        # if no file whatsoever, bool_runall is True to run everything without loading
        if not bool_sourceinfo and not bool_targetinfo and not bool_agent3 and not bool_agent4:
            print(f"{i} running every agent")
            bool_runall = True

        # prepare necessary files
        source_video_idx = source_idx_list[i]
        source_action_sequence, scenegraphnotused = agent_init.get_video_info(source_video_idx)
        source_goal_category, source_goal_description = agent_init.get_source_video_metadata(source_video_idx)
        source_scene_graph = agent_init.extract_spatial_context(source_spatial_json_list[i])
        target_scene_graph = agent_init.extract_spatial_context(target_spatial_json_list[i])
        
        source_uid  = source_spatial_json_list[i]['video_id']
        target_uid = target_spatial_json_list[i]['video_id']
        source_file_name = target_spatial_json_list[i]['source_file_name']
        target_file_name = target_spatial_json_list[i]['target_file_name']
        target_equal_ratio  = target_spatial_json_list[i]['target_equal_ratio']
        trial_index  = target_spatial_json_list[i]['trial_index']

        #Format raw dictionary scene graph into string. action sequence is already a single string, so no worries
        source_scene_graph = llmfuncs.format_scene_graph(source_scene_graph)
        target_scene_graph = llmfuncs.format_scene_graph(target_scene_graph)

        # sourceinfo and targetinfo
        while not bool_sourceinfo and not bool_targetinfo:
            print("saving info")
            with open(PATH_SOURCEINFO, 'wb') as f:
                print(f"{i} ")
                dict = {
                    "idx": i, 
                    "source_idx": source_video_idx, 
                    "source_uid": source_uid, 
                    "source_file_name": source_file_name, 
                    "target_equal_ratio": target_equal_ratio, 
                    "trial_index": trial_index, 
                    "source_goal_category": source_goal_category,
                    "source_goal_description": source_goal_description,
                    "source_action_sequence": source_action_sequence,
                    "source_scene_graph": source_scene_graph,
                    }
                pickle.dump(dict, f)
                bool_sourceinfo = True

            with open(PATH_TARGETINFO, 'wb') as f:
                print(f"{i} ")
                dict = {
                    "idx": i,
                    "target_idx": (source_video_idx+10)%100, 
                    "target_uid": target_uid, 
                    "target_file_name": target_file_name,
                    "target_equal_ratio": target_equal_ratio, 
                    "trial_index": trial_index,                     
                    "target_scene_graph": target_scene_graph, 
                    }
                pickle.dump(dict, f)
                bool_targetinfo = True
              
        # -----------------------
        # AGENT1: PREDICT CORE ACTIVITY FOR LATER
        # -----------------------       


        if bool_agent1a:
            with open(PATH_AGENT1a, 'rb') as f:
                source_core_activity = pickle.load(f)
        else:
            while not bool_agent1a:
                try:
                    response_1a = llmfuncs.run_agent1a_llm_norag(
                        source_action_sequence,
                        source_scene_graph,
                        agent_api_name,
                        agent_model_name,
                        temperature =0.5
                    )
                    print(response_1a)
                    source_core_activity = str(response_1a)
                    print(f"stringed {source_core_activity}")

                    with open(PATH_AGENT1a, 'wb') as f:
                        pickle.dump(source_core_activity, f)        
                        print(f"sgent1a saved: source_core_activity")
                        bool_agent1a = True

                except Exception as e:
                    print(f"Agent1a failed at index {i}: {e}")
                    continue

        # -----------------------
        # AGENT3: PREDICT ACTION SEQUENCE: ONLY SPATIAL EXAMPLE
        # -----------------------
        if bool_agent3:
            #load and be done with it
            with open(PATH_AGENT3, 'rb') as f:
                target_action_sequence = pickle.load(f)           
        else:
            while not bool_agent3:    
                try:         
                    response_3 = llmfuncs.run_agent3_llm_s2s(
                        source_action_sequence,
                        source_scene_graph,
                        target_scene_graph,
                        agent_api_name,
                        agent_model_name,
                        temperature =0.5                        
                    )
                    target_action_sequence = response_3
                    target_action_sequence = str(target_action_sequence)

                    print(f"source seq {source_action_sequence}")
                    print(f"3b output {target_action_sequence}")

                    with open(PATH_AGENT3, 'wb') as f:
                        pickle.dump(target_action_sequence, f)        
                        print(f"agent3 saved: target_action_sequence")
                        bool_agent3 = True

                except Exception as e:
                    print(f"Agent3 failed at index {i}: {e}")
                    continue

        
        # print(target_action_sequence)            
        # #TODO: NEED CORE ACTIVITY TO DO FILTER CHECK
        # # -----------------------
        # # FINAL: TEST FAITHFULNESS TO CORE ACTIVITY
        # # -----------------------        
        # if bool_agent4:
        #     with open(PATH_AGENT4, 'wb') as f:
        #         final_response = pickle.load(f)
        # else:
        #     while not bool_agent4:
        #         try:
        #             final_response = run_core_action_test(
        #                 source_core_activity,
        #                 target_activity_taxonomy,
        #                 target_action_sequence,
        #                 agent_api_name,
        #                 agent_model_name
        #             )

        #             print(final_response)
        #             with open(PATH_AGENT4, 'wb') as f:
        #                 pickle.dump(final_response, f)        
        #                 print(f"agent4 saved: final answer {final_response}")
        #                 bool_agent4 = True      

        #         except Exception as e:
        #             print(f"Agent4 failed at index {i}: {e}")
        #             continue  