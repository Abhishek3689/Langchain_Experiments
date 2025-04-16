from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation',
    temperature=.7
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(template="Tell me about {player} in 40 words",input_variables=['player'])

while True:
    print("if want to exit input quit")
    user_input=input("Enter the player name :")
    if user_input=='quit':
        break
    chain=LLMChain(llm=model,prompt=prompt)
    response=chain.invoke(input={'player':user_input})
    print(response)