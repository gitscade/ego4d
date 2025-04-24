'''
This baseline is an Activity Matching System consisting of Three Agentic RAG components
1. activity recognition agent*
2. activity transfer agent*
3. action sequence prediction agent*

*agent may or may not use spatial context dataset for improved accuracy
Agents are chained
**when using OLLAMA, turn on ollama with code in workflow_ollama.ipynb
'''
import sys
import os
from langchain.schema.runnable import RunnableLambda
# packages
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
import 02_agent.agent_database as agent_database
import 02_agent.agent_input as agent_input
import 02_agent.agent_query as agent_query
import 02_agent.agent_prompt as agent_prompt
import 02_agent.workflow_data as workflow_data
import 02_agent.workflow_agent1a as agent1
import 02_agent.workflow_agent2a as agent2a
import 02_agent.workflow_agent2b as agent2b
import workflow_agent3 as agent3


def run_agents_serial(input_text):
    result1 = agent1.run_agent1(input_text)
    result2a = agent2a.run_agent2(result1)
    result2b = agent2b.run_agent2(result2a)
    result3 = agent3.run_agent3(result2b)
    return result3

def run_agent_chain(input_text):
    runnable_agent1 = RunnableLambda(lambda x: agent1.run(x))
    runnable_agent2a = RunnableLambda(lambda x: agent2a.run(x))
    runnable_agent2b = RunnableLambda(lambda x: agent2b.run(x))
    runnable_agent3 = RunnableLambda(lambda x: agent3.run(x))
    agent_pipeline = runnable_agent1 | runnable_agent2a | runnable_agent2b | runnable_agent3    
    return agent_pipeline.invoke(input_text)

if __name__ == "__main__":

    source_idx = int(input("source scene idx: "))
    target_idx = int(input("target scene idx: "))
    source_spatial_video = workflow_data.spatial_test_video_list[source_idx]
    target_spatial_video = workflow_data.spatial_test_video_list[target_idx]
    source_scene_graph = agent_input.extract_spatial_context(source_spatial_video)
    target_scene_graph = agent_input.extract_spatial_context(target_spatial_video)

    #TODO Define Agent and run?
