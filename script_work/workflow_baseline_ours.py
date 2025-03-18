'''
This baseline is an Activity Transfer System consisting of Three Agentic RAG components
1. activity recognition agent*
2. activity transfer agent*
3. action sequence prediction agent*

*agent uses spatial context dataset for improved accuracy.

Agent is chained (recommended by GPT4o)
'''
import sys
import os
sys.path.append(os.path.abspath('/root/project')) # add root path to sys.path


import script_work.workflow_agent3 as agent3




# from langchain.schema.runnable import RunnableLambda

# # Convert each agent into a Runnable
# runnable_agent1 = RunnableLambda(lambda x: agent1.run(x))
# runnable_agent2 = RunnableLambda(lambda x: agent2.run(x))
# runnable_agent3 = RunnableLambda(lambda x: agent3.run(x))

# # Chain them together
# agent_pipeline = runnable_agent1 | runnable_agent2 | runnable_agent3

# # Run the pipeline
# output = agent_pipeline.invoke("Start process")
# print(output)
