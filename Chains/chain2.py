import os
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation',
    temperature=0.5,
    max_new_tokens=100
)

model=ChatHuggingFace(llm=llm)
output_parser=StrOutputParser()

template='tell me about {player} in 2 lines.Description should include country and number of matches'
prompt=PromptTemplate(template=template,input_variables=['player'])

chain=LLMChain(llm=model,prompt=prompt,output_parser=output_parser)
memory={}
def store_info(key,value):
    memory[key]=value

def retrieve_info(key):
    return memory.get(key,None)

while True:
    user_input=input("Enter the Player Name enter quit or exit to exit ethe chat:")
    if user_input in ('quit','exit'):
        break
    res=chain.run(player=user_input)
    store_info(user_input,res)
    print(res)
print('*******************************************************************')
print("Thank you for using Chatbot")
print('*******************************************************************')
print("\nMemory : ",memory)