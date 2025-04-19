from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import getpass
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#import torch
from src.utils import save_model

# Load the .env file
load_dotenv()



llm=HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation'
)

model = ChatHuggingFace(
    llm=llm
)

save_model(model,"artifacts/model.pth")