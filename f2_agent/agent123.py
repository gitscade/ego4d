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
import f1_init.database_init as database_init
import f2_agent.agent_prompt as agent_prompt

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
         """You are a helpful taxonomy summarizer that summarizes an action_sequence in input scene_graph, using tools. State your final answer in a section labeled 'Final Answer:'.:
        
            Final Answer: [Your answer]
            
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
        ("system", """You are a helpful classifier that constructs a taxonomy of an activity in a scene. If you finalized the taxonomy, you must reorder the taxonomy using appropriate tool. Return the final re-ordered taxonomy following the format below. USE SAME TAXONOMY FORMAT WITHOUT ADDING ANYTHING MORE:
            
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
         
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: ["query": {query}, "source_action_sequence": {source_action_sequence}, "source_scene_graph": {source_scene_graph}, "source_core_activity": {source_core_activity}]            
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

             For example, if input is called "cook steak", you are to create a taxonomy for the classification of the steak. If steak involves cooking with salary and mustard sauce, in the source_action sequence, you can create taxonomy in a dictionary form like:

             {
             "main ingredient": pork
             "preparation method": roasted
             "garnish": "salary"
             "sauce": "mustard"
             "spice":"salted"
             }
             
             As you see, there are 5 levels of key-value pairs. You will decide the name of the key in the taxonomy dictionary, and the corresponding value for the key. The key and the value pair for the input noun must be accomplished by the result of the source_action_sequence. For example, if steak is said to use only meat, main ingredient should be meat. If you are making dough, you will consider appropriate key names, not blindly following the key for the steak example.
             """},
            {"role": "user", "content": f"Here is the query:\n{query}\n"},
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the core activity:\n{source_core_activity}\n"}
    ]

    MESSAGE_REORDER_TAXONOMY = [
            {"role": "system", "content": """You are a taxonomy reorderring expert, that reorders 5-Leval classification taxonomy so that higher concept key values apperas first. I will give you an example.
             
             First, you receive a taxonomy in dictionary form like this:

             {
             "sauce": "mustard"
             "preparation method": "roasted"
             "main ingredient": "pork"             
             "garnish": "salary"
             "spice":"salted"
             }

             You also see the input of source_core_activity which is "Cook Steak", and you can reason that the taxonomy is about the classification of Steak cooked by source_action_sequence, in a scene of source_scene_graph.

             Let's say you are explaining this steak to a person in a sentence. You are explaining what this steak is: "This is a pork steak, that is roasted, with salary garnish. We applied mustard sauce and salted it." This explanation gives information about the identity of the steak, stating key-value pair that is more essential in the definition of this steak. This leads to re-arranged dictionary taxonomy like. Make no mistake. This explanation is about the identity of the steak, not the sequence of how it is made.

             {
             "main ingredient": "pork"   
             "preparation method": "roasted" 
             "garnish": "salary"
             "sauce": "mustard"
             "spice":"salted"
             }

             Use this reasoning to re-order the elements in the dictionary and return it as output.
             """},
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the core activity:\n{source_core_activity}\n"}
    ]


    return AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION, MESSAGE_REORDER_TAXONOMY

def get_agent2a_message(inputlist:list):
    """
    func: returns prompt and messages used for agent2a\n
    input: single list: [tools, sequence, scenegraph, target_scene_graph, source_activity_taxonomy]\n
    return: AGENT2a_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputlist[0]
    source_action_sequence = inputlist[1]
    source_scene_graph = inputlist[2]
    target_scene_graph = inputlist[3]    
    source_activity_taxonomy = inputlist[4]
    query = "Construct a common taxonomy that is applicable for both source and target scene graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT2a_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful taxonomy examiner. Your task is to examine whether the input taxonomy for the source_scene_graph is applicable target_scene_graph with common_taxonomy_prediction_tool and return the output taxonomy tool as the common taxonomy. Return the final re-ordered taxonomy following the format below. USE SAME TAXONOMY FORMAT WITHOUT ADDING ANYTHING MORE:    
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: [Appropriate Input for Tool]
        """),

        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the source scene graph: {source_scene_graph}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )


    MESSAGE_COMMON_TAXONOMY_PREDICTION = [
            {"role": "system", "content": """You are a taxonomy examiner that checks if each level in taxonomy can be achieved in the target scene, and returns a common taxonomy.
             
             First, you receive a source_activity_taxonomy in dictionary form:

             {
             "main ingredient": "pork"   
             "preparation method": "roasted" 
             "garnish": "salary"
             "sauce": "mustard"
             "spice":["salted","peppered"]
             }

             For each key-value pair (starting from-"main ingredient":"pork"-for this example), you check whether each value can be acquired from target_scene_graph or any possible actions in target_scene_graph. If value is possible, the key-value pair is unchanged. If no, replace the value in the key-value pair to "empty". Doing this for all key-value pairs will result in example similar to the dictionary below:

             {
             "main ingredient": "pork"   
             "preparation method": "roasted" 
             "garnish": "empty"
             "sauce": "empty"
             "spice":["salted","empty"]
             }

             In this case, this means there is no entity of "salary" for garnish and "mustard" for sauce in the target_scene_graph, or that no activity in the target_scene_graph can generate entity of "salary" or "garnish", respectively, for garnish or sauce key. Hence, the values for these keys are changed to "empty". For spice, pepper was not available so, only "pepper" becomes "empty".

             Return the final dictionary of key-values as output. This dictionary, following the above logic, and have changed values or retain the same information as the input. You must use the same format as the example to represent the output dictionary."""}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"}
        ]
    return AGENT2a_PROMPT, MESSAGE_COMMON_TAXONOMY_PREDICTION

def get_agent2b_message(inputs:list):
    """
    func: returns prompt and messages used for agent2b\n
    input: single list: [tools, sequence, scenegraph, target_scenegraph, source_taxonomy, common_taxonomy]\n
    return: AGENT2b_PROMPT, MESSAGE_ACTIVITY_PREDICTION
    """
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]
    source_activity_taxonomy = inputs[4]
    common_activity_taxonomy = inputs[5]
    query = "construct a target_acvity_taxonomy that is applicable to target_scene_graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT2b_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful taxonomy enricher that fills in a missing class of an input taxonomy with appropriate values based on target_scene_graph, using tools. Return the final re-filled taxonomy as a target_activity_taxonomy, following the format below. USE SAME TAXONOMY FORMAT WITHOUT ADDING ANYTHING MORE:    
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: [Appropriate Input for Tool]
        """),

        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the source scene graph: {source_scene_graph}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "This is the source activity taxonomy: {common_activity_taxonomy}"),        
        ("system", "Available tools: {tools}. Actively use retrieval tools to get a plausible answer."),
        ("system", "Tool names: {tool_names}"),
        ("user", "This is the user query: {query}"),
        ("assistant", "{agent_scratchpad}")
        ]
        )

    MESSAGE_TARGET_TAXONOMY_PREDICTION = [
            {"role": "system", "content": """You are a taxonomy generator for target_scene_graph that checks common_activity_taxonomy and converts it to target_activity_taxonomy. Let me give you a step-by-step example of how you function.
             
             First, you receive a source_activity_taxonomy in dictionary form:

             {
             "main ingredient": "pork"   
             "preparation method": "roasted" 
             "garnish": "empty"
             "sauce": "empty"
             "spice":["salted","empty"]
             }

             For each key-value pair that holds value "empty" (in this case, values for "garnish" and "sauce"), you must find suitable alternative entity in the target_scene that can fill these values. 

             Let's say source_activity_taxonomy values for "garnish", "sauce", "spaice" are "salary", "mustard", and ["salted", "peppered"], respectively. You are to find the most similar substitute for these in the target_scene_graph.

             If target_scene_graph has "chill bottle" as entity, this can act as similar substitute to add "chilli" to the spice. Therefore, "empty" for spice becomes "chillied". In the same manner, if substitute vegetables for "garnish" is found in target_scene as "potato" "onion", you can pick the most vegetable that performs similar function as salary. Since salarly has crunchy texture, "onion" can be a substitue for "salary" in the source_acitivty_taxonomy's "garnish". When no sauce whatsoever is found in the scene, the "sauce" value should remain "empty". Overall, this will result in a new taxonomy below:

            {
             "main ingredient": "pork"   
             "preparation method": "roasted" 
             "garnish": "onion"
             "sauce": "empty"
             "spice":["salted","chillied"]
             }

             Following the logic above, you will come up with a new taxonomy for the target_scene_graph. Return this final dictionary of key-values as output. You must use the same format as the example to represent the output dictionary."""}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{common_activity_taxonomy}\n"},            
        ]    
    
    
    return AGENT2b_PROMPT, MESSAGE_TARGET_TAXONOMY_PREDICTION

def get_agent3_message(inputs:list):
    tools = inputs[0]
    source_action_sequence = inputs[1]
    source_scene_graph = inputs[2]
    target_scene_graph = inputs[3]
    source_activity_taxonomy = inputs[4]
    target_activity_taxonomy = inputs[5]
    query = "predict target_action_sequence that can realize the target_activity_taxonomy in the target_scene_graph"
    tool_names =", ".join([t.name for t in tools])    

    AGENT3_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful action planner that constructs an action sequence for the target_scene_graph to achieve the target_activity_taxonomy, using tools. Return the final action_sequence as a target_action_sequence, following the format below. USE SAME FORMAT AS THE source_action_sequence WITHOUT ADDING ANYTHING MORE:
        
            Final Answer: [Your answer]
            
        Otherwise, use this format for step-by-step answering. First, explain your reasinging in 'Thought:'. Then, explain used tool in a section labeled 'Action:'. Finally, print out the input to pass on to the tool in a section labeled as 'Action Input:', following the format below. Retrieve relevant information with retriever tools first to gather similar examples.:
        
            Thought: [Your reasoning]
            Action: [Tool name]
            Action Input: [appropriate input for the tool]
        """),

        ("system", "This is the user action sequence in source scene: {source_action_sequence}."),
        ("system", "This is the source scene graph: {source_scene_graph}."),
        ("system", "This is the target scene graph: {target_scene_graph}."),
        ("system", "This is the source activity taxonomy: {source_activity_taxonomy}"),
        ("system", "This is the source activity taxonomy: {target_activity_taxonomy}"),        
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
             "main ingredient": "pork"   
             "preparation method": "roasted" 
             "garnish": "onion"
             "sauce": "empty"
             "spice":["salted","chillied"]
             }

             

             Let's say source_activity_taxonomy values for "garnish", "sauce", "spaice" are "salary", "mustard", and ["salted", "peppered"], respectively. You are to find the most similar substitute for these in the target_scene_graph.

             If target_scene_graph has "chill bottle" as entity, this can act as similar substitute to add "chilli" to the spice. Therefore, "empty" for spice becomes "chillied". In the same manner, if substitute vegetables for "garnish" is found in target_scene as "potato" "onion", you can pick the most vegetable that performs similar function as salary. Since salarly has crunchy texture, "onion" can be a substitue for "salary" in the source_acitivty_taxonomy's "garnish". When no sauce whatsoever is found in the scene, the "sauce" value should remain "empty". Overall, this will result in a new taxonomy below:

            {
             "main ingredient": "pork"   
             "preparation method": "roasted" 
             "garnish": "onion"
             "sauce": "empty"
             "spice":["salted","chillied"]
             }

             Following the logic above, you will come up with a new taxonomy for the target_scene_graph. Return this final dictionary of key-values as output. You must use the same format as the example to represent the output dictionary."""}, 
            {"role": "user", "content": f"Here is the source_action_sequence:\n{source_action_sequence}\n" },
            {"role": "user", "content": f"Here is the source scene graph:\n{source_scene_graph}\n"},
            {"role": "user", "content": f"Here is the target scene graph:\n{target_scene_graph}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{source_activity_taxonomy}\n"},
            {"role": "user", "content": f"Here is the source activity taxonomy:\n{target_activity_taxonomy}\n"},            
        ]   

    AGENT3_PROMPT = ChatPromptTemplate.from_messages([])
    MESSAGE_TARGET_SEQUENCE_PREDICTION = []
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

def activity_prediction(MESSAGE_ACTIVITY_PREDICTION):
    """Predict an core summary of the user activity based on input"""
    try:
        #OPENAI
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = MESSAGE_ACTIVITY_PREDICTION,
            temperature=0.5
        )      
        response = response.choices[0].message.content.strip()
        #OLLAMA
        # response = ollama.chat(
        #     model = agent_init.LLM_MODEL_4MINI,
        #     messages= MESSAGE_ACTIVITY_PREDICTION,
        #     options={ 'temperature':0.5 }
        # )
        # response = response['message']['content']
        return response   

    except Exception as e:
        return f"Error: activity_prediction: {str(e)}"

def predict_activity_taxonomy(MESSAGE_TAXONOMY_CREATION):
    """Predict an core summary of the user activity based on input"""
    try:
        #OPENAI
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model = "gpt-4",
            messages = MESSAGE_TAXONOMY_CREATION,
            temperature=0.5
        )      
        response = response.choices[0].message.content.strip()     
        return response   

    except Exception as e:
        return f"Error: activity_prediction: {str(e)}"

def reorder_activity_taxonomy(unordered_taxonomy):
    """order an activity taxonomy"""
    try:
        #OPENAI
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model = "gpt-4",
            messages = MESSAGE_REORDER_TAXONOMY,
            temperature=0.5
        )      
        response = response.choices[0].message.content.strip()
        #OLLAMA
        # response = ollama.chat(
        #     model = agent_init.LLM_MODEL_4MINI,
        #     messages= MESSAGE_ORDER_TAXONOMY,
        #     options={ 'temperature':0.5 }
        # )
        # response = response['message']['content'] 
        return response   

    except Exception as e:
        return f"Error: activity_prediction: {str(e)}"

def make_common_taxonomy(MESSAGE_COMMON_TAXONOMY_PREDICTION):
    """Test if goal of deep activity can be met in current target_scene."""

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=agent_init.LLM_MODEL_GPT4MINI,
        messages= MESSAGE_COMMON_TAXONOMY_PREDICTION,
        temperature=0.5
    )
    executability = response.choices[0].message.content.strip()
    return f"Thought: Executability.\n{executability}"

def make_target_activity_taxonomy(MESSAGE_TARGET_TAXONOMY_PREDICTION):
    """Make Target Taxonomy"""

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=agent_init.LLM_MODEL_GPT4MINI,
        messages= MESSAGE_COMMON_TAXONOMY_PREDICTION,
        temperature=0.5
    )
    executability = response.choices[0].message.content.strip()
    return f"Thought: Executability.\n{executability}"



# -----------------------
# Tool GET Funcs
# -----------------------
def get_agent1a_tools():
    """
    return tools
    """
    tools=[
    Tool(
        name = "goalstep_retriever_tool",
        func = goalstep_information_retriever,
        description = "Retrieves relevant activity step information in other scenes."
    ),
    Tool(
        name = "spatial_retriever_tool",
        func = spatial_information_retriver,
        description = "Retrieves relevant scene and entity information for similar environments, where state changes of entities takes place in spatiotemporal fashion."
    ),
    Tool(
        name = "activity_prediction_tool",
        func = lambda _: activity_prediction(MESSAGE_ACTIVITY_PREDICTION),
        description = "Activity prediction tool, which can summarize the sequential multiple actions into a single verb and a single noun."
    ),
    Tool(
        name = "reorder_activity_taxonomy_tool",
        func = lambda _: reorder_activity_taxonomy(MESSAGE_REORDER_TAXONOMY),
        description = "reorder_activity_taxonomy, which can reorder the input taxonomy."
    ),    
    ]
    return tools

def get_agent1b_tools():
    """
    return tools
    """
    tools = [
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
        name = "predict_activity_taxonomy_tool",
        func = lambda _: predict_activity_taxonomy(MESSAGE_TAXONOMY_CREATION),
        description = "This tool generates a spefic and deep hierarchical description of source activity"
    ),
    ]
    return tools

def get_agent2a_tools():
    """
    return tools
    """
    tools = [
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
            func = lambda _: make_common_taxonomy(MESSAGE_COMMON_TAXONOMY_PREDICTION),
            description = "Test if goal of deep activity can be met in current target_scene."
        )
    ]
    return tools

def get_agent2b_tools():
    """
    return tools
    """
    tools = [
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
    return tools

def wip_get_agent3_tools():
    return []

# -----------------------
# Agent Function
# -----------------------
def run_agent_1a(input):
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
        llm=agent_init.LLM_MODEL_GPT4MINI,
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

def run_agent_1b(input):
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
        llm=agent_init.LLM_MODEL_GPT4MINI,
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

def run_agent_2a(input):
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
        llm=agent_init.LLM_MODEL_GPT4MINI,
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

def run_agent_2b(input):
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

    TOOLNAMES =", ".join([t.name for t in TOOLS])
    QUERY = "Find Target Activity Taxonomy."    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_init.LLM_MODEL_GPT4MINI,
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
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

def run_agent3():
    """"
    func: run agent 3\n
    input: [tools_3, AGENT3_PROMPT, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, target_activity_taxonomy]\n
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

    TOOLNAMES =", ".join([t.name for t in TOOLS])
    QUERY = "Predict Action Sequence for the target scene grph"    
    MEMORY = ConversationBufferWindowMemory(k=3, input_key="query") # only one input key is required fo this!

    AGENT = create_react_agent(
        tools=TOOLS,
        llm=agent_init.LLM_MODEL_GPT4MINI,
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
            "tools": TOOLS,
            "tool_names": TOOLNAMES,
            "agent_scratchpad": "" 
         },
        config={"max_iterations": 5}
    )
    return response

if __name__ == "__main__":

    # -----------------------
    # FILES / FORMATTING
    # -----------------------
    source_video_idx = 1
    target_video_idx = 2

    source_goalstep_video = database_init.goalstep_test_video_list[source_video_idx]
    source_action_sequence = agent_init.extract_lower_goalstep_segments(source_goalstep_video)
    source_action_sequence = json.dumps(source_action_sequence)
    
    source_spatial_video = database_init.spatial_test_video_list[source_video_idx]
    source_scene_graph = agent_init.extract_spatial_context(source_spatial_video)
    source_scene_graph = json.dumps(source_scene_graph, indent=2)

    target_spatial_video = database_init.spatial_test_video_list[target_video_idx]
    target_scene_graph = agent_init.extract_spatial_context(target_spatial_video)
    target_scene_graph = json.dumps(target_scene_graph, indent=2)
    
    # -----------------------
    # PREDICT CORE ACTIVITY
    # -----------------------
    tools_1a = get_agent1a_tools()
    input_1a_message = [tools_1a, source_action_sequence, source_scene_graph]
    AGENT1a_PROMPT, MESSAGE_ACTIVITY_PREDICTION = get_agent1a_message(input_1a_message)

    input_1a_agent = [tools_1a, AGENT1a_PROMPT, source_action_sequence, source_scene_graph]
    response_1a = run_agent_1a(input_1a_agent)

    # -----------------------
    # PREDICT FULL ACTIVITY TAXONOMY
    # -----------------------
    source_core_activity = response_1a['output']
    tools_1b = get_agent1b_tools()
    input1b_message = [tools_1b, source_action_sequence, source_scene_graph, source_core_activity]
    AGENT1b_PROMPT, MESSAGE_TAXONOMY_CREATION, MESSAGE_REORDER_TAXONOMY = get_agent1b_message(input1b_message)

    input_1b_agent = [tools_1b, AGENT1b_PROMPT, source_action_sequence, source_scene_graph, source_core_activity]
    response_1b = run_agent_1b(input_1b_agent)
    print(response_1b)

    # # -----------------------
    # # PREDICT COMMON ACTIVITY TAXONOMY
    # # -----------------------    
    source_activity_taxonomy = response_1b['output']
    tools_2a = get_agent2a_tools()
    # # tools, sequence, scenegraph, target_scenegraph, source_activity_taxonomy]
    input2a_message = [tools_2a, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy]
    AGENT2a_PROMPT, MESSAGE_COMMON_TAXONOMY_PREDICTION=get_agent2a_message(input2a_message)
    input2a_agent = [tools_1b, AGENT2a_PROMPT, source_action_sequence, source_scene_graph, source_core_activity]
    response_2a = run_agent_2a()

    # # -----------------------
    # # PREDICT TARGET ACTIVITY TAXONOMY
    # # -----------------------    

    # # MOVE UP ACTIVITY IS NOT FOUND
    # common_activity_taxonomy = response_2a['output']
    # tools_2b = get_agent2b_tools()
    # # tools, sequence, scenegraph, target_scenegraph, target_scene_graph, source_activity_taxonomy, common_activity_taxonomy]
    # PROMPT2b, MESSAGE_TARGET_TAXONOMY_PREDICTION=get_agent2b_message(tools_2a, source_action_sequence, source_scene_graph, target_scene_graph, source_activity_taxonomy, common_activity_taxonomy)
    # response_2b = run_agent_2b()

    # # # -----------------------
    # # # PREDICT TARGET ACTION SEQUENCE
    # # # -----------------------
    # # target_activity_taxonomy = response_2b['output']

    # # response_3 = []

    # # # -----------------------
    # # # Save stuff for evaluation
    # # # -----------------------
    # # target_action_sequence = response_3['output']
    # # input_outputs = [source_action_sequence, source_core_activity, source_activity_taxonomy, common_activity_taxonomy, target_activity_taxonomy, target_action_sequence]

