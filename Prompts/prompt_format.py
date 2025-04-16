from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

template="Can you tel me about {country} in {number} sentence"

input_variables=['country','number']

prompt=PromptTemplate(
    template=template,
    input_variables=input_variables,
    validate_template=True
)

final_prompt=prompt.format(country='India',number=5)
print(final_prompt)