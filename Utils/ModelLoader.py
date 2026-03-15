from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


class modelLoader():
  def __init__(self, modelName):
    self.modelName = modelName
     
  def load_llm(self):
    print("LLM loading...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    print(f"Model Name: {self.modelName}")
    llm=ChatGroq(model=self.modelName, api_key=groq_api_key)
    return llm