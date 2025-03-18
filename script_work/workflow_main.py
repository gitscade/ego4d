'''
This baseline is an Activity Matching System consisting of Three Agentic RAG components
1. activity recognition agent*
2. activity transfer agent*
3. action sequence prediction agent*

*agent may or may not use spatial context dataset for improved accuracy
Agents are chained
'''
# from langchain.schema.runnable import RunnableLambda
import sys
import os
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path
import workflow_agent1 as agent1
import workflow_agent2 as agent2
import workflow_agent3 as agent3


def chain_agents(input_text):
    result1 = agent1.run_agent1(input_text)
    result2 = agent2.run_agent2(result1)
    result3 = agent3.run_agent3(result2)
    return result3

if __name__ == "__main__":
    final_output = chain_agents("Start process")
    print("Final Output:", final_output)

# # Convert each agent into a Runnable
# runnable_agent1 = RunnableLambda(lambda x: agent1.run(x))
# runnable_agent2 = RunnableLambda(lambda x: agent2.run(x))
# runnable_agent3 = RunnableLambda(lambda x: agent3.run(x))

# Chain them together
# agent_pipeline = runnable_agent1 | runnable_agent2 | runnable_agent3

# # Run the pipeline
# output = agent_pipeline.invoke("Start process")
# print(output)
