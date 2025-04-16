from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import getpass
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace



# Load the .env file
load_dotenv()



llm=HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation'
)

model = ChatHuggingFace(
    llm=llm
)

response = model.invoke("How many states are there in USA?Give me numbers only")
print(response.content)