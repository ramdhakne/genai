# tested by Ram Dhakne
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np

api_key = "XYZJqvKRsCWy6AgSfXlZaDQyhnT1u1OzMLOPS"
client = MistralClient(api_key=API_KEY)


def embed_sim_check():

    # embedding check
    sentences = ["AI is transforming the world.", "Machine learning is a subset of AI."]
    response = client.embeddings(model="mistral-embed", input=sentences)

    # Compute Cosine Similarity 
    vec1 = np.array(response.data[0].embedding)
    vec2 = np.array(response.data[1].embedding)

    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"Cosine Similarity: {similarity}")

def streaming_chat_res():
    # streaming chat response check
    messages = [ChatMessage(role="user", content="Tell me a joke.")]

    response = client.chat(model="mistral-large-latest", messages=messages, stream=True)

    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)

def main():
    embed_sim_check()
    #streaming_chat_res()

if __name__ == "__main__":
    main()