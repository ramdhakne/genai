from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
import time
from getpass import getpass

# execute the program with python3.9 rag_basic.py

# works on following software
# brew install python@3.9
# pip3.9 install faiss-cpu==1.7.4 mistralai

#api_key= getpass("SJqvKRsCWy6AgSfXlZaDQyhnT1u1HzML")
api_key = "SJqvKRsCWy6AgSfXlZaDQyhnT1u1HzML"
client = Mistral(api_key=api_key)

#print("client", client)

# for a getting embeddings
def get_text_embedding(input):
    #print("inside func client=", client)
    time.sleep(2)
    print("inside func api_key=", api_key)
    #client = Mistral(api_key=api_key)
    print("NEW inside func client=", client)
    embeddings_batch_response = client.embeddings.create(
          model="mistral-embed",
          inputs=input
      )
    return embeddings_batch_response.data[0].embedding

def run_mistral(user_message, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)

# get the data
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text

f = open('eassy.txt', 'w')
f.write(text)
f.close()

print("len of the essay =", len(text))

# chunk the essay
chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# test count for chunks

print("Number of chunks of size for essay =", len(chunks))

# gen embeddings
text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
print("text embeddings shape", text_embeddings.shape)
print("text embeddings", text_embeddings)

# Load into a vector database
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

question = "What were the two main things the author worked on before college?"
question_embeddings = np.array([get_text_embedding(question)])

print("question_embeddings.shape=", question_embeddings.shape)
print("question_embeddings", question_embeddings)

# Retrieve similar chunks from the vector database 
D, I = index.search(question_embeddings, k=2)
print(I)

retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

print("Printing retrived chunks from vector DB\n", retrieved_chunk)

time.sleep(2)

# creating the prompt
prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

res = run_mistral(prompt)

print("\n\n****response****\n\n\n\n", res)