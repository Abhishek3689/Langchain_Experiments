import os
#from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation',
    temperature=.7
)

template=PromptTemplate(template='Tell me about {country} in {number} lines',
                      input_variables=['country'])

country=input("enter the country name")
number=input("enter the number of lines to demostrate :")
prompt=template.invoke({'country':country,'number':number})

model=ChatHuggingFace(llm=llm)
response=model.invoke(prompt)

print(response.content)
