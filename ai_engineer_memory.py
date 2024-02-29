from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import ShellTool
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import datetime
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

# Initialize OpenAI and Chroma
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TABLE_NAME = "experiments_feb_3"#"experiments"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name=TABLE_NAME)

# Other initializations and configurations
shell_tool = ShellTool()
tools = [PythonREPLTool(), shell_tool]

def save_memory(task_text):
    temp_file_path = "temp_text_file.txt"
    with open(temp_file_path, "w") as file:
        file.write(task_text)
    loader = TextLoader(temp_file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f"docs: ", docs)
    Chroma.from_documents(docs, embeddings, collection_name=TABLE_NAME, persist_directory="./chroma_db")
    os.remove(temp_file_path)

def query_memory(task_description, num_results):
    """
    Queries the Chroma DB for similar past tasks based on the provided task description.
    
    :param task_description: A string description of the current task.
    :param num_results: The number of similar past tasks to retrieve.
    :return: A list of similar past tasks and their details, where the similarity score is above 0.75.
    """ 
    results = chroma_db.similarity_search_with_score(task_description, k=num_results)
    #print(results)
    relevant_tasks = []
    for result, score in results:
        if score < 0.35:
            relevant_tasks.append(result.page_content)
    return relevant_tasks

def format_context(context_array):
    context_formatted = "\n*******RELEVANT CONTEXT******\n"
    if len(context_array) > 0:
        for context in context_array:
            context_formatted += f"\n{context}\n"
    else:
        context_formatted += f"\nNONE\n"
    context_formatted += "\n*******END CONTEXT******\n"
    return context_formatted

#save_memory("a bear walked down the street")
#relevant_tasks = query_memory("the model was fine-tuned and evaluated over 5 epochs.", 3)
#print(format_context(relevant_tasks))

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

agent = create_openai_functions_agent(ChatOpenAI(temperature=0, model="gpt-4-1106-preview", openai_api_key=OPENAI_API_KEY), tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, max_iterations=3)

task = f"""Write an object detection neural network in pytorch that takes a image as input
and outputs bounding boxes of where the animals are in the image. The animals are hiding in camouflage.
The data for training can be found in './data/camouflaged_animals'. In that folder there is a folder named 'images' containing 386 images for training.
There's also a json file named 'annotations.json' containing the metadata with the bounding box labels. The json is in the COCO format.
Feel free to either fine-tune a model from the pytorch hub or train something de novo.
The data in the data folder can be used as is and doesn't have to be copied, a backup of it already exists.
Train the network for 5 epochs and return the precision and recall metrics of the final network. """
relevant_tasks = query_memory(task, 1)
context = format_context(relevant_tasks)
task_w_context = task + context


#print(relevant_tasks)
#print(f"context: ", context)
#print("")
#print(f"task_w_context: ", task_w_context)

with get_openai_callback() as cb:
    response = agent_executor.invoke(
        {
            "input": task_w_context
        }
    )
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    summary = f"Date: {current_date}\nTask: {task}\nResult: {response['output']}"
    save_memory(summary)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

