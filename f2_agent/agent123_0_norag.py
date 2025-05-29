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
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}, "source_action_sequence": {source_action_sequence}, "source_scene_graph": {source_scene_graph}]

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

def get_agent1b_message(inputs:list):
    """
    func: returns prompt and messages used for agent1a\n
    input: single list: [tools, sequence, scenegraph, coreactivity]\n
    return: AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    source_core_activity = inputs[3]
    query = "make a N=5 level taxonomy for the given core activity"
    tool_names =", ".join([t.name for t in tools])    

    AGENT1b_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful classifier that constructs a taxonomy of an activity in a scene. 
            
         Use the following format for each step:
         
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: [The input to the action]

         When you have enough information, conclude with: Final Answer. AGENT MUST NOT MODIFY TAXONOMY IN ANY WAY. Just Print out the final answer in this format without adding anything:
            
            Final Answer: [Your answer]
        """),
        ("system", "This is the user action sequence: {source_action_sequence}."),
        ("system", "Predicted activity must be able to be performed in this scene: {source_scene_graph}."),
        ("system", "Core activity of the current scene: {source_core_activity}"),
        ("system", "Available tools: {tools}. Actively use retrieval tools to come up with plausible answer."),
        ("system", "Tool names: {tool_names}"),  # for React agents
        ("user", "This is the query for the agent: {query}"),
        ("assistant", "{agent_scratchpad}")  # for React agents
        ])


    MESSAGE_TAXONOMY_CREATION = [
            {"role": "system", "content": """You are a taxonomy constructer, that makes 5-Leval classification taxonomy for a given noun in the input.
             
             First, you receive a pair of words: a verb and a noun. This is called core activity. 
             
             From information from source_action_sequence, source_scene_graph, and retrieved relevant information, you will construct a more detailed taxonomy of the input noun in the core activity.

             For example, if input is called "cook steak", you are to create a taxonomy for the classification of the steak. Taxonomy should answer question of "What is the cooked steak?" I can give you an example of taxonomy for steak below. The keys-values here are just example, and you will choose yours so that your taxonomy best represent activity in source_action_sequence and source_scene_graph:

             {
             "sauce": "mustard",
             "preparation method": "roasted",
             "main ingredient": "pork",             
             "garnish": "salary",
             "spice":"salted",
             }
             
             As you see, there are 5 levels of key-value pairs. ONE WORD for EACH KEY. You will decide the name of the key in the taxonomy dictionary, and the corresponding value for the key. The key and the value pair for the input noun must be accomplished by the result of the source_action_sequence. For example, if steak is said to use only meat, main ingredient should be meat. If you are making dough, you will consider appropriate key names, not blindly following the key for the steak example. Only assign ONE OBJECT for EACH KEY.

             You also see the input of source_core_activity which is "Cook Steak", and you can reason that the taxonomy is about the classification of Steak cooked by source_action_sequence, in a scene of source_scene_graph.

             Let's say you are explaining this steak to a person in a sentence. You are explaining what this steak is: "This is a pork steak, that is roasted, with salary garnish. We applied mustard sauce and salted it." This explanation gives information about the identity of the steak, stating key-value pair that is more essential in the definition of this steak. This leads to re-arranged dictionary taxonomy like. Make no mistake. This explanation is about the identity of the steak, not the sequence of how it is made.
             
             
             ```json
             {
             "main ingredient": "pork",   
             "preparation method": "roasted", 
             "garnish": "salary",
             "sauce": "mustard",
             "spice":"salted",
             }
             ```

             Use this reasoning to re-order the elements in the dictionary and return it as output. ONLY CHANGE THE ORDER OF THE INPUT TAXONOMY IN THIS STEP and finalize the answer. STRICTLY FOLLOW THE FORMAT ABOVE for final answer.

             """},
            {"role": "user", "content": f"Here is the query:\n{query}\n"},
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the core activity:\n{source_core_activity}\n"}
    ]

    # MESSAGE_REORDER_TAXONOMY = [
    #         {"role": "system", "content": """You are a taxonomy reorderring expert, that reorders 5-Leval classification taxonomy so that higher concept key values apperas first. I will give you an example.
             
    #          First, you receive a taxonomy in dictionary form like this:

    #          {
    #          "sauce": "mustard",
    #          "preparation method": "roasted",
    #          "main ingredient": "pork",             
    #          "garnish": "salary",
    #          "spice":"salted",
    #          }

    #          You also see the input of source_core_activity which is "Cook Steak", and you can reason that the taxonomy is about the classification of Steak cooked by source_action_sequence, in a scene of source_scene_graph.

    #          Let's say you are explaining this steak to a person in a sentence. You are explaining what this steak is: "This is a pork steak, that is roasted, with salary garnish. We applied mustard sauce and salted it." This explanation gives information about the identity of the steak, stating key-value pair that is more essential in the definition of this steak. This leads to re-arranged dictionary taxonomy like. Make no mistake. This explanation is about the identity of the steak, not the sequence of how it is made.

    #          {
    #          "main ingredient": "pork",   
    #          "preparation method": "roasted", 
    #          "garnish": "salary",
    #          "sauce": "mustard",
    #          "spice":"salted",
    #          }

    #          Use this reasoning to re-order the elements in the dictionary and return it as output. YOU CAN ONLY CHANGE THE ORDER OF THE INPUT TAXONOMY. DO NOT MODIFY THE VALUE.
    #          """},
    #         {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
    #         {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"},
    #         {"role": "user", "content": f"Here is the core activity:\n{source_core_activity}\n"},
    # ]
    return AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION #, MESSAGE_REORDER_TAXONOMY

def get_agent2a_message(inputs:list):
    """
    func: returns prompt and messages used for agent2a\n
    input: single list: [tools, sequence, scenegraph, target_scene_graph, source_activity_taxonomy]\n
    return: AGENT2a_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]    
    source_activity_taxonomy = inputs[4]
    query = "Construct a common taxonomy that is applicable for both source and target scene graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT2a_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful taxonomy examiner. Your task is to examine whether the input taxonomy for the source_scene_graph is applicable target_scene_graph with common_taxonomy_prediction_tool and return the output taxonomy tool as the common taxonomy. Return the final taxonomy following the format below. AGENT MUST NOT MODIFY TAXONOMY FROM THE TOOLS IN ANY WAY. JUST PASS THE TAXONOMY FROM THE TOOL WITHOUT OMITTING OR CHANGING ANYTHING.   
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}]
        """),

        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )
        # ("system", "This is the source scene graph: {source_scene_graph}."),

    MESSAGE_COMMON_TAXONOMY_PREDICTION = [
            {"role": "system", "content": """You are a taxonomy examiner that checks if each key value in taxonomy can be acquired in the target_scene_graph, and returns a common taxonomy.
             
             First, you receive a source_activity_taxonomy in dictionary form. Here's an example:

             {
             "main ingredient": "pork",
             "preparation method": "roasted", 
             "garnish": "salary",
             "sauce": "mustard",
             "spice":["salted","peppered"]
             }

             For each key-value pair, starting from "main ingredient":"pork", you check the target_scene_graph or any possible actions in target_scene_graph. If value for the key exists in the target_scene_graph, the key-value pair is unchanged. If no, replace the value in the key-value pair to "empty". Do not retain value not inside or acquireable in scene graph! Doing this for all key-value pairs will result in example similar to the dictionary below:

             ```json
             {
             "main ingredient": "empty",   
             "preparation method": "roasted", 
             "garnish": "empty",
             "sauce": "empty",
             "spice":["salted","empty"],
             }
             ```

             In this example, there is no entity of "pork", "salary", and "mustard" inside the target_scene_graph, or that no activity in the target_scene_graph can generate entity of "pork", "salary" or "garnish". Hence, the values for these keys are changed to "empty". For spice, only salt was available, so "pepper" becomes "empty". 

             This changed dictionary is called common_activity_taxonomy. In constructing this taxonomy, use the same format as the example. DO NOT DELETE  OR REFORMAT ANY KEY while returning common_activity_taxonomy. Encase the final answer with ```json, and ``` symbols to return output:
             
            Thought: Your reasoning for each of the five key
            Action: common_activity_taxonomy                  
             """}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"}
        ]
    
            # {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},    
    return AGENT2a_PROMPT, MESSAGE_COMMON_TAXONOMY_PREDICTION

def get_agent2b_message(inputs:list):
    """
    func: returns prompt and messages used for agent2b\n
    input: single list: [tools, sequence, scenegraph, target_scenegraph, source_taxonomy, common_taxonomy, core_activity]\n
    return: AGENT2b_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]
    source_activity_taxonomy = inputs[4]
    common_activity_taxonomy = inputs[5]
    source_core_activity = inputs[6]
    query = "construct a target_acvity_taxonomy that is applicable to target_scene_graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT2b_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful taxonomy enricher that fills in a missing class of an input taxonomy with appropriate values based on target_scene_graph, using tools.        
         
         Return the final re-filled taxonomy as a target_activity_taxonomy, following the format below. AGENT MUST NOT MODIFY TAXONOMY FROM THE TOOLS IN ANY WAY. JUST PASS THE TAXONOMY FROM THE TOOL WITHOUT OMITTING OR CHANGING ANYTHING.
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}, "target_scene_graph": {target_scene_graph}]
        """),

        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the source scene graph: {source_scene_graph}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "This is the source common activity taxonomy: {common_activity_taxonomy}"),
        ("system", "This is the source core activity: {source_core_activity}"),  
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )
    
    MESSAGE_TARGET_TAXONOMY_EXAMINER = [
            {"role": "system", "content": """You are a taxonomy examiner for a generated target_activity_taxonomy. You're main task is to examine if target_activity_taxonomy follows the context of source_core_acivitiy. Let me give you a step-by-step example of how you function.
             
             First, you receive a target_activity_taxonomy in dictionary form:

            {
             "main ingredient": "pork",
             "preparation method": "roasted", 
             "garnish": "salary",
             "sauce": "mustard",
             "spice":["salted","peppered"]
             }

             You also receive source_core_activity. For this example suppose you have:

             "cook steak"

             For this example, the target_activity_taxnomy can be used for core activity of "cook steak". You will simple return the target_activity_taxonomy.
            
             While for this example, we found candidates for filling in the empty fields, we could encounter target_scene_graph where fulfilling source_core_activity is impossible. In this case, fill EVERY VALUE of the target_activity_taxonomy with "impossible" to make sure target_activity_taxonomy is impossible. 

             DO NOT DELETE OR REFORMAT ANY KEY in the common_activity_taxnomy dictionary in the target_activity_taxonomy"""}, 
   
            {"role": "user", "content": f"Here is the source_core_activity:\n{source_core_activity}\n" },            
        ]    
    
            # {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},            

            # {"role": "user", "content": f"Here is the source scene graph. This is NOT target scene graph:\n{source_scene_graph}\n"},

    MESSAGE_TARGET_TAXONOMY_PREDICTION = [            
        {"role": "system", "content": """You are a taxonomy generator for target_scene_graph that checks common_activity_taxonomy and converts it to target_activity_taxonomy. Let me give you a step-by-step example of how you function.
             
             First, you receive a source_activity_taxonomy in dictionary form:

             {
             "main ingredient": "empty",   
             "preparation method": "roasted", 
             "garnish": "empty",
             "sauce": "empty",
             "spice":["salted","empty"],
             }

             You also receive source_core_activity and source_activity_taxonomy. For this example suppose you have:

             "cook steak"

             {
             "main ingredient": "pork",
             "preparation method": "roasted", 
             "garnish": "salary",
             "sauce": "mustard",
             "spice":["salted","peppered"]
             }

             For each key-value pair that holds value "empty" (in this case, values for "garnish" and "sauce"), you must find suitable alternative entity in the target_scene that can fill these values, in the context of source_core_activity.

             First print out the entities in the target_scene_graph. ONLY USE THESE entities in filling out the "empty" values in common_activity_taxonomy:

             target_scene_graph_entities: [items in target_scene_graph]

             Let's say for the main ingredient, there is "chicken" in the target_scene. "chicken" is not the same "main ingredient" as "pork". But under the context of "cook steak" "chicken" can be a main ingredient. Therefore "chicken" in the target_scene_graph can fill in as the "main ingredient". If "tofu" is there instead of "chicken", if tofu steak is a possible dish, you can put in "tofu" as main ingredient. If there is only "lettunce", maybe making a steak is just impossible and "main ingredient" has to remain "empty".

             Let's say source_activity_taxonomy values for "garnish", "sauce", "spaice" are "salary", "mustard", and ["salted", "peppered"], respectively. You are to find the most similar substitute for these in the target_scene_graph.

             If target_scene_graph has "chill bottle" as entity, this can act as similar substitute to add "chilli" to the spice. Therefore, "empty" for spice becomes "chillied". 
             
             In the same manner, if substitute vegetables for "garnish" is found in target_scene as "potato" "onion", you can pick the most vegetable that performs similar function as salary. Since salarly is there as garnish for steak, this is mostly for its texture. In this reagard, "onion" can be a substitue for "salary" in the source_acitivty_taxonomy's "garnish". If there is no onion in target, then, anything that serves as garnish like potato can be used to replace the "empty" value. When no sauce whatsoever is found in the scene, the "sauce" value should remain "empty". Overall, this will result in a new taxonomy below:

             ```json
             {
             "main ingredient": "pork",   
             "preparation method": "roasted", 
             "garnish": "onion",
             "sauce": "empty",
             "spice":["salted","chillied"]
             }
             ```

             While for this example, we found candidates for filling in the empty fields, we could encounter target_scene_graph where fulfilling source_core_activity is impossible. In this case, fill the "empty" field with "impossible" to make sure target_activity_taxonomy is impossible. 

             Following the logic above, you will come up with a new taxonomy for the target_scene_graph. Return this final dictionary of key-values as output. You must use the same format enclosed in the ``` to represent the final answer. DO NOT DELETE OR REFORMAT ANY KEY in the common_activity_taxnomy dictionary in the target_activity_taxonomy"""}, 
   
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the source_core_activity:\n{source_core_activity}\n" },            
            {"role": "user", "content": f"Here is the common activity taxonomy:\n{common_activity_taxonomy}\n"},     
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
        ]


    #MESSAGE_TARGET_TAXONOMY_EXAMINER is not used
    return AGENT2b_PROMPT, MESSAGE_TARGET_TAXONOMY_PREDICTION, MESSAGE_TARGET_TAXONOMY_EXAMINER

def get_agent3_message(inputs:list):
    """
    get agent3 inputs
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]
    source_activity_taxonomy = inputs[4]
    target_activity_taxonomy = inputs[5]
    source_core_activity = inputs[6]
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
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "This is the target activity taxonomy: {target_activity_taxonomy}"),        
        ("system", "This is the source core activity: {source_core_activity}"),        
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )

    MESSAGE_TARGET_SEQUENCE_PREDICTION = [
            {"role": "system", "content": """You are a action planner expert that makes action sequence in a target_scene_graph to achieve the taxonomy values of the target_activity_taxonomy. Let me give you a step-by-step example of how you function.
             
             First, you receive a target_activity_taxonomy in dictionary form:

            {
             "main ingredient": "pork",   
             "preparation method": "roasted", 
             "garnish": "onion",
             "sauce": "empty",
             "spice":["salted","chillied"]
             }

             Let's also suppose that you receive a source_core_activity and source_activity_taxonomy as follows:

             "Cook Steak"

             Therefore, you can see that the target_activity_taxonomy is describing the noun ("steak") after it is "cooked" in the target_scene. 

             When looking at all information from source_scene_graph, source_action_sequence, and source_activity_taxonomy you find that value for each key is started and completed at certain steps at source_action_sequence. For example, in a source_activity_taxonomy like below, you will know that salt or salt bottle is present at the scene, but the spice "salted" value only can be completed when player or user in the scene performs a state of salting the ingredient or steak in the source_action_sequence. In the same logic, trying to achieve the "salted" value in "spice" key can start when player tries to grab salt/salt bottle or performs salting action. "pork" key is already started and completed in a step where it is chosen as main ingredient.

            
             {
             "main ingredient": "pork",   
             "preparation method": "roasted", 
             "garnish": "salary",
             "sauce": "mustard",
             "spice":["salted","peppered"]
             }


             Following the logic above, you can make action sequence for the target_scene and target_activity_taxonomy. First, you should make a logical action_sequence so that each value for the key in the target_activity_taxonomy is achieved, and the core_activity is realized with the resources and entities available in the target_scene_graph. Print your logic in format below:

             unordered target taxonomy: your action sequence for target taxonomy
             Logic: your logic for each key in the target taxonomy
             
             Then, a re-ordering tool should be called, so that each value for the key is started and completed at similar absolute step, or relative step with respect to steps for the corresponding keys in the source_action_sequence. This will return the final re-ordered target_action_sequence.
             
             Finally, you will return the target_action_sequence as output. The format of the target_scene_graph should follow the format of the source_scene_graph. STRICTLY Follow the format below to print the output:
             
             Final Answer: ["action1", "action2", ..., "final action]
             """}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{target_activity_taxonomy}\n"},
            {"role": "user", "content": f"Here is the source core activity:\n{source_core_activity}\n"},            
        ]   
    return AGENT3_PROMPT, MESSAGE_TARGET_SEQUENCE_PREDICTION
#------------------------
#Tool Funcs
#------------------------
def goalstep_information_retriever(query:str):
    """Retrieve the most relevant goalstep dataset documents based on a user's query."""
    context = agent_init.goalstep_retriever.invoke(query)
    return f"User Query: {query}. similar goalstep examples: {context}" 

def spatial_information_retriver(query:dict):
    """Retrieve the most relevant spatial context documents based on a user's query"""
    context = agent_init.spatial_retriever.invoke(query)
    return f"User Query: {query}. similar spatial examples: {context}"

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

def predict_activity_taxonomy(MESSAGE_TAXONOMY_CREATION, TOOL_LLM_API, TOOL_LLM_STR):
    """Predict an core summary of the user activity based on input"""
    my_temperature = 0.5
    try:
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_TAXONOMY_CREATION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_TAXONOMY_CREATION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    

    except Exception as e:
        return f"Error: activity_prediction: {str(e)}"

def make_common_taxonomy(MESSAGE_COMMON_TAXONOMY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR):
    """Test if goal of deep activity can be met in current target_scene."""
    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_COMMON_TAXONOMY_PREDICTION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_COMMON_TAXONOMY_PREDICTION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: common_taxonomy_prediction: {str(e)}"

def make_target_activity_taxonomy(MESSAGE_TARGET_TAXONOMY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR):
    my_temperature = 0.5
    try:        
        if TOOL_LLM_API == "openai":
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = MESSAGE_TARGET_TAXONOMY_PREDICTION,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= MESSAGE_TARGET_TAXONOMY_PREDICTION,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: target_taxonomy_prediction: {str(e)}"
  
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

def get_agent1b_tools():
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
        name = "predict_activity_taxonomy_tool",
        func = lambda _: predict_activity_taxonomy(MESSAGE_TAXONOMY_CREATION, TOOL_LLM_API, TOOL_LLM_STR),
        description = "This tool generates a taxonomical description of source action sequence"
    ),
    # Tool(
    #     name = "reorder_activity_taxonomy_tool",
    #     func = lambda _: reorder_activity_taxonomy(MESSAGE_REORDER_TAXONOMY),
    #     description = "This tool reorders activity taxnomy taxonomy"
    # ),
    ]
    return tools

def get_agent2a_tools():
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
            name = "make_common_taxonomy_tool",
            func = lambda _: make_common_taxonomy(MESSAGE_COMMON_TAXONOMY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR),
            description = "Test if goal of deep activity can be met in current target_scene."
        )
    ]
    return tools

def get_agent2b_tools():
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
            name = "make_target_activity_taxonomy_tool",
            func = lambda _: make_target_activity_taxonomy(MESSAGE_TARGET_TAXONOMY_PREDICTION, TOOL_LLM_API, TOOL_LLM_STR),
            description = "make target activty taxonomy that achieves core activity in the target scene."
        ),    
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

def run_agent_1b(input, agent_llm_chat):
    """"
    func: run agent 1a with source video idx info\n
    input: [tools_1b, AGENT1b_PROMPT, source_action_sequence, source_scene_graph, source_core_activity]\n
    output: response
    """
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    source_core_activity= input[4]
    TOOLNAMES =", ".join([t.name for t in TOOLS])

    QUERY = "Categorically describe source activity in a very specific way."    
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
            "source_core_activity": source_core_activity,
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

def run_agent_2a(input, agent_llm_chat):
    """"
    func: run agent 2a\n
    input: [tools_2a, AGENT2a_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy]\n
    output: common_activity_taxonomy
    """    
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    target_scene_graph = input[4]
    source_activity_taxonomy= input[5]

    TOOLNAMES =", ".join([t.name for t in TOOLS])
    QUERY = "Find Common Activity Taxonomy."    
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
            "source_activity_taxonomy": source_activity_taxonomy,
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

def run_agent_2b(input, agent_llm_chat):
    """"
    func: run agent 2b\n
    input: [tools_2b, AGENT2b_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, common_activity_taxnomy]\n
    output: target_activity_taxonomy
    """    
    # Load input
    TOOLS = input[0]
    AGENT_PROMPT = input[1]
    source_action_sequence = input[2]
    source_scene_graph = input[3]
    target_scene_graph = input[4]
    source_activity_taxonomy= input[5]
    common_activity_taxonomy= input[6]
    source_core_activity = input[7]

    TOOLNAMES =", ".join([t.name for t in TOOLS])
    QUERY = "Find Target Activity Taxonomy."    
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
            "source_activity_taxonomy": source_activity_taxonomy,
            "common_activity_taxonomy": common_activity_taxonomy,  
            "source_core_activity": source_core_activity,          
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

def run_tool_2c(input, TOOL_LLM_API, TOOL_LLM_STR):
    """
    input: [{"main object": "empty", "container": "empty", "status": "empty", "type": "empty", "category": "empty"}, hand clothes]
    
    """
    print(input)
    my_temperature = 0.5
    
    TOOL_2c_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a taxonomy examiner for a generated target_activity_taxonomy. You're main task is to examine if target_activity_taxonomy follows the context of source_core_acivitiy. Let me give you a step-by-step example of how you function.
            
            First, you receive a target_activity_taxonomy in dictionary form:

            {
            "main ingredient": "pork",
            "preparation method": "roasted", 
            "garnish": "salary",
            "sauce": "mustard",
            "spice":["salted","peppered"]
            }

            You also receive source_core_activity. For this example suppose you have:

            "cook steak"

            For this example, the target_activity_taxnomy can be used for core activity of "cook steak". You will simple return the target_activity_taxonomy. In this case, only return this information

            {
            "main ingredient": "pork",
            "preparation method": "roasted", 
            "garnish": "salary",
            "sauce": "mustard",
            "spice":["salted","peppered"]
            }
        
            While for this example, we found candidates for filling in the empty fields, we could encounter target_scene_graph where fulfilling source_core_activity is impossible. In this case, fill EVERY VALUE of the target_activity_taxonomy with "impossible" to make sure target_activity_taxonomy is impossible. If our example happens to become liek this, you will only return the final answer like this

            {
            "main ingredient": "impossible",
            "preparation method": "impossible", 
            "garnish": "impossible",
            "sauce": "impossible",
            "spice": "impossible"
            }

            Only return the revised or same target_activity_taxonomy as answer. DO NOT include anything more.
        """),


        ("system", "This is the source core activity: {source_core_activity}"),  
        ("system", "This is the target activity taxonomy: {target_activity_taxnomy}"),
        ]
        )


    try:        
        if TOOL_LLM_API == "openai":
            messages = TOOL_2c_PROMPT.format_messages(
                source_core_activity=input['source_core_activity'],
                target_activity_taxnomy=str(input['target_activity_taxnomy'])  # ensure it's stringified
            )

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model = TOOL_LLM_STR,
                messages = messages,
                temperature=my_temperature
            )      
            return response.choices[0].message.content.strip()
        elif TOOL_LLM_API == "ollama":
            response = ollama.chat(
                model = TOOL_LLM_STR,
                messages= TOOL_2c_PROMPT,
                options={ 'temperature':my_temperature }
            )
            return response['message']['content']    
    except Exception as e:
        return f"Error: subtool_target_taxonomy_prediction: {str(e)}"    

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
    source_activity_taxonomy= input[5]
    target_activity_taxonomy= input[6] 
    source_core_activity = input[7]

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
            "source_activity_taxonomy": source_activity_taxonomy,
            "target_activity_taxonomy": target_activity_taxonomy, 
            "source_core_activity": source_core_activity,           
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

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

def test_idx(source_idx_list):
    import csv
    test_list = []
    for i in range(len(source_idx_list)):
        source_video_idx = source_idx_list[i]        
        target_video_idx = (source_video_idx + 10) % 71

        source_action_sequence1, source_scene_graph1, source_id1, source_id2 = agent_init.get_video_info_idxtest(source_video_idx)
        source_scene_graph = source_spatial_json_list[i]

        target_scene_graph1 = database_init.spatial_test_video_list[target_video_idx]
        target_scene_graph = target_spatial_json_list[i]
                
        print(f"{source_video_idx} {target_video_idx} {source_id1} {source_scene_graph['video_id']} {target_scene_graph1['video_id']} {target_scene_graph['video_id']} {target_scene_graph['spatial_similarity']} ")

        test_list.append([source_video_idx, target_video_idx, source_id1, source_scene_graph['video_id'], target_scene_graph1['video_id'], target_scene_graph['video_id'], target_scene_graph['spatial_similarity']])

    with open("spatial_similarity_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["source_idx", "target_idx", "source_scene_id1", "source_scene_id1", "target_scene_id1", "target_scene_id", "spatial_similarity"])
        writer.writerows(test_list)

if __name__ == "__main__":
    # -----------------------
    # TEST SETTINGS
    # -----------------------
    agent_api_name = "openai"
    agent_model_name = "gpt-4o"
    tool_api_name = "openai"
    tool_model_name = "gpt-4o"    
    AGENT_LLM_API, AGENT_LLM_STR, AGENT_LLM_CHAT = agent_init.SET_LLMS(agent_api_name, agent_model_name, temperature=0.2)
    TOOL_LLM_API, TOOL_LLM_STR, TOOL_LLM_CHAT = agent_init.SET_LLMS(tool_api_name, tool_model_name, temperature=0.2)

    # SETUP FIRST INPUTS
    PATH_SOURCE_TARGET_INPUT = constants_init.PATH_SOURCE_TARGET + "/input/source_target_video_list.pkl"
    with open(PATH_SOURCE_TARGET_INPUT, "rb") as f:
        source_target_list = pickle.load(f)
    BASELINE_FOLDER = "/output-norag/"
    PATH_SOURCE_TARGET_OUTPUT = constants_init.PATH_SOURCE_TARGET + BASELINE_FOLDER

    # Get scenegraph for source and target
    PATH_AUGv2 = constants_init.PATH_AUGMENTATION + "TESTSET_Augmented_Data_v2/"
    source_spatial_json_list, target_spatial_json_list, aug_levels = agent_init.get_source_target_spatial_json_list_augv2(PATH_AUGv2)
    # Make source_idx_list that matches length of the above json list
    source_idx_list = [i for i in range(len(source_spatial_json_list)//len(aug_levels)) for _ in range(len(aug_levels))]

    # test idx for input so that source-target pair is matched perfectly
    # test_idx(source_idx_list)

    # # for i in range(0, len(source_list)):
    # for i in range(len(source_idx_list)):
    for i in range(295, 296):

        PATH_SOURCEINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_sourceinfo.pkl"
        PATH_TARGETINFO = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_targetinfo.pkl"
        PATH_AGENT1a = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent1a.pkl"
        PATH_AGENT1b = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent1b.pkl"
        PATH_AGENT2a = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent2a.pkl"
        PATH_AGENT2b = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent2b.pkl"
        PATH_AGENT3 = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent3.pkl"
        PATH_AGENT4 = PATH_SOURCE_TARGET_OUTPUT + f"pair{i}_agent4.pkl"  

        source_video_idx = source_idx_list[i]
        source_action_sequence, scenegraphnotused = agent_init.get_video_info(source_video_idx)
        source_scene_graph = agent_init.extract_spatial_context(source_spatial_json_list[i])
        target_scene_graph = agent_init.extract_spatial_context(target_spatial_json_list[i])
        source_uid  = source_spatial_json_list[i]['video_id']
        target_uid = target_spatial_json_list[i]['video_id']
        spatial_similarity  = target_spatial_json_list[i]['spatial_similarity']

        # # -----------------------
        # # AGENT1a: PREDICT CORE ACTIVITY
        # # -----------------------
        with open(PATH_SOURCEINFO, 'wb') as f:
            dict = {"source_idx": source_video_idx, "source_uid": source_uid, "source_action_sequence": source_action_sequence, "source_scene_graph": source_scene_graph, "spatial_similarity": spatial_similarity}
            pickle.dump(dict, f)        
            print(f"SOURCE INFO: {i} {source_video_idx} {source_uid} {source_action_sequence} {spatial_similarity}")   
        with open(PATH_TARGETINFO, 'wb') as f:
            dict = {"target_idx": (source_video_idx+10)%71, "target_uid": target_uid, "target_scene_graph": target_scene_graph}
            pickle.dump(dict, f)
            # print(f"TARGET INFO: {i} {(source_video_idx+10)%71} {target_uid} {target_action_sequence} {target_scene_graph}")
            
        try:
            tools_1a = get_agent1a_tools()
            input_1a_message = [tools_1a, source_action_sequence, source_scene_graph]
            AGENT1a_PROMPT, MESSAGE_ACTIVITY_PREDICTION = get_agent1a_message(input_1a_message)
            input_1a_agent = [tools_1a, AGENT1a_PROMPT, source_action_sequence, source_scene_graph]
            response_1a = run_agent_1a(input_1a_agent, AGENT_LLM_CHAT)
            source_core_activity = response_1a['output']

            with open(PATH_AGENT1a, 'wb') as f:
                pickle.dump(source_core_activity, f)        
                print(f"agent1a saved: source_core_activity")   

        except Exception as e:
            print(f"Agent1a failed at index {i}: {e}")
            continue


        # -----------------------
        # AGENT1b: PREDICT FULL ACTIVITY TAXONOMY
        # -----------------------
        try:        
            tools_1b = get_agent1b_tools()
            input1b_message = [tools_1b, source_action_sequence, source_scene_graph, source_core_activity]
            # AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION, MESSAGE_REORDER_TAXONOMY = get_agent1b_message(input1b_message)
            AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION = get_agent1b_message(input1b_message)
            input_1b_agent = [tools_1b, AGENT1b_PROMPT, source_action_sequence, source_scene_graph, source_core_activity]
            response_1b = run_agent_1b(input_1b_agent, AGENT_LLM_CHAT)
            source_activity_taxonomy = response_1b['output']
            print(f"1b output {source_activity_taxonomy}")
            
            source_activity_taxonomy = re.sub(r"^```json\s*|\s*```$", "", source_activity_taxonomy.strip())
            source_activity_taxonomy = util_funcs.jsondump_agent_response(source_activity_taxonomy)

            with open(PATH_AGENT1b, 'wb') as f:
                pickle.dump(source_activity_taxonomy, f)        
                print(f"agent1b saved: source_activity_taxonomy")   

        except Exception as e:
            print(f"Agent1b failed at indiex {i}: {e}")
            continue


        # -----------------------
        # AGENT2a: PREDICT COMMON ACTIVITY TAXONOMY
        # -----------------------            
        # print(source_activity_taxonomy)
        try:  
            
            tools_2a = get_agent2a_tools()
            input2a_message = [tools_2a, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy]
            AGENT2a_PROMPT, MESSAGE_COMMON_TAXONOMY_PREDICTION=get_agent2a_message(input2a_message)
            input2a_agent = [tools_2a, AGENT2a_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy]
            response_2a = run_agent_2a(input2a_agent, AGENT_LLM_CHAT)
            common_activity_taxonomy = response_2a['output']
            print(f"2a output {common_activity_taxonomy}")

            common_activity_taxonomy = re.sub(r"^```json\s*|\s*```$", "", common_activity_taxonomy.strip())
            common_activity_taxonomy = util_funcs.jsondump_agent_response(common_activity_taxonomy)

            with open(PATH_AGENT2a, 'wb') as f:
                pickle.dump(common_activity_taxonomy, f)        
                print(f"agent2a saved: common_activity_taxonomy")   

        except Exception as e:
            print(f"Agent2a failed at index {i}: {e}")
            continue


        # -----------------------
        # AGENT2b: PREDICT TARGET ACTIVITY TAXONOMY
        # -----------------------    
        try:  
            
            tools_2b = get_agent2b_tools()
            input2a_message = [tools_2b, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, common_activity_taxonomy, source_core_activity]    
            AGENT2b_PROMPT, MESSAGE_TARGET_TAXONOMY_PREDICTION, MESSAGE_TARGET_TAXONOMY_EXAMINER=get_agent2b_message(input2a_message)
            input2b_agent = [tools_2b, AGENT2b_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, common_activity_taxonomy, source_core_activity]    
            response_2b = run_agent_2b(input2b_agent, AGENT_LLM_CHAT)
            target_activity_taxonomy = response_2b['output']
            print(f"2b output {target_activity_taxonomy}")

            target_activity_taxonomy = re.sub(r"^```json\s*|\s*```$", "", target_activity_taxonomy.strip())
            target_activity_taxonomy = util_funcs.jsondump_agent_response(target_activity_taxonomy)
            
            with open(PATH_AGENT2b, 'wb') as f:
                pickle.dump(target_activity_taxonomy, f)        
                print(f"agent2b saved: target_activity_taxonomy saved")   

        except Exception as e:
            print(f"Agent2b failed at index {i}: {e}")
            continue
        
        # -----------------------
        # PREDICT TARGET ACTION SEQUENCE
        # -----------------------
        try:                      
            tools_3 = get_agent3_tools()
            input3_message = [tools_3, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, target_activity_taxonomy, source_core_activity]
            AGENT3_PROMPT, MESSAGE_TARGET_SEQUENCE_PREDICTION=get_agent3_message(input3_message)       
            input3_agent = [tools_3, AGENT3_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, target_activity_taxonomy, source_core_activity]   
            response_3 = run_agent3(input3_agent, AGENT_LLM_CHAT)
            target_action_sequence = response_3['output']

            # SERIALIZE TO FORMAT
            target_action_sequence = re.sub(r"^```json\s*|\s*```$", "", target_action_sequence.strip())
            target_action_sequence = util_funcs.jsondump_agent_response(target_action_sequence)

            with open(PATH_AGENT3, 'wb') as f:
                pickle.dump(target_action_sequence, f)        
                print(f"agent3 saved: target_action_sequence")

        except Exception as e:
            print(f"Agent3 failed at index {i}: {e}")
            continue        



        # -----------------------
        # FINAL: TEST FAITHFULNESS TO CORE ACTIVITY
        # -----------------------        
        try:
            final_response = run_core_action_test(
                source_core_activity,
                target_activity_taxonomy,
                target_action_sequence,
                agent_api_name,
                agent_model_name
            )

            print(final_response)
            with open(PATH_AGENT4, 'wb') as f:
                pickle.dump(final_response, f)        
                print(f"agent3 saved: output: final answer {final_response}")       
            print(f"AGENT4")

        except Exception as e:
            print(f"Agent4 failed at index {i}: {e}")
            continue                        