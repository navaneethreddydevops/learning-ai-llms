import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
load_dotenv()

llm = OpenAI(temparetaure=0.6)
name = llm("I wan to use python for AI")
print(name)