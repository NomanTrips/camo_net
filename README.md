# Camonet - detecting animals in natural images

## Project Overview
This project aims to develop an object detection model specialized in identifying animals in camouflage within natural images. The project includes a script
to train an RCNN on this task and also some langchain agent scripts to have GPT-4 write and train a network for the same task.

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- langchain

### Installation
1. Clone the repository
2. Download the dataset and put it into the /data folder.
3. Modify the scripts to point to where you downloaded the data.
2. Create a .env file in the directory containing your OPENAI_API_KEY (if you're using the agent scripts).

### Usage
- Fintune a basic RCNN from the pytorch hub on the task: python finetune_rcnn.py
- Have GPT-4 write and train a network to perform the task: python ai_engineer.py

## Warning!
The agent scripts "ai_engineer.py" and "ai_engineer_memory.py" can consume a lot of tokens. Lower 'max_iterations' in them if necessary.

### License
MIT

