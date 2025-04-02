import os
import time
import requests
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

# works only
# brew install python@3.9

# python3.9 rag_langchain.py
# pip3.9 install llama-index==0.10.55 llama-index-llms-mistralai==0.1.18 llama-index-embeddings-mistralai mistralai==0.4.2

api_key = "XYZJqvKRsCWy6AgSfXlZaDQyhnI1111LLMLLMLLM"

# get the data
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text

f = open('essay.txt', 'w')
f.write(text)
f.close()

print("len of the essay =", len(text))

# Load data
reader = SimpleDirectoryReader(input_files=["essay.txt"])
documents = reader.load_data()
# Define LLM and embedding model
Settings.llm = MistralAI(model="mistral-medium", api_key=api_key)
Settings.embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=api_key)
# Create vector store index
index = VectorStoreIndex.from_documents(documents)
# Create query engine
query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query(
    "What were the two main things the author worked on before college?"
)
print("\n\n********Response********\n")
print(str(response))


