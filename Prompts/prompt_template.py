import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
load_dotenv()

llm=HuggingFaceEndpoint
