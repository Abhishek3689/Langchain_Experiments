from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
import torch

load_dotenv()
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# First try this simpler version without ChatHuggingFace
try:
    # Load model and tokenizer directly
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.7,
        max_new_tokens=100
    )
    
    # Test basic generation first
    result = pipe("Tell me a joke about AI")[0]["generated_text"]
    print("Direct pipeline test:", result)
    
    # Then try with LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    model = ChatHuggingFace(llm=llm)
    result = model.invoke("Tell me a joke about AI")
    print("Chat version:", result.content)

except Exception as e:
    print(f"Error: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Run 'pip install sentencepiece protobuf'")
    print("2. Check your internet connection")
    print("3. Try with CPU-only: remove device_map and add device='cpu'")
    print("4. Try a smaller model like 'distilgpt2' first")