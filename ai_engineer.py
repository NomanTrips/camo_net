from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import ShellTool
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

shell_tool = ShellTool()

tools = [PythonREPLTool(), shell_tool]

instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
You have access to a sandboxed linux environment to run the code on. This is availible via the 'Shell' tool.
The sandboxed linux instance has significant compute resources, including a RTX 3090 with 24 GB VRAM.
The linux instance also has popular machine learning tools installed including python, pytorch and CUDA.
Be careful about running commands that could generate a lot of output in the shell because this information is being fed back to you and we don't want it to clog up the context window accidently.
When using ls in large directories, pipe its output to head or tail (e.g., ls | head -n 10) to limit displayed results and maintain system performance.
For large files, use head or tail with cat (e.g., cat filename.json | head -n 20) to view a specific portion and avoid loading the entire file, optimizing resource use and response time.
The Shell tool output will be truncated if there's too much text in it.
"""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_openai_functions_agent(ChatOpenAI(temperature=0, model="gpt-4-1106-preview", openai_api_key=api_key), tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, max_iterations=20)

with get_openai_callback() as cb:
    response = agent_executor.invoke(
        {
            "input": """ Write an object detection neural network in pytorch that takes a image as input
            and outputs bounding boxes of where the animals are in the image. The animals are hiding in camouflage.
            The data for training can be found in './data/camouflaged_animals'. In that folder there is a folder named 'images' containing 386 images for training.
            There's also a json file named 'annotations.json' containing the metadata with the bounding box labels. The json is in the COCO format.
            Feel free to either fine-tune a model from the pytorch hub or train something de novo.
            The data in the data folder can be used as is and doesn't have to be copied, a backup of it already exists.
            Train the network for 5 epochs and return the precision and recall metrics of the final network. """
        }
    )
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")