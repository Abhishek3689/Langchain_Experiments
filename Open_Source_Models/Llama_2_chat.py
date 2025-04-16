from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import os
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-2-7b-chat',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

res=model.invoke("who is the current president of USA")
print(res)