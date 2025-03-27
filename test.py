# tested by Ram Dhakne
from mistralai import Mistral

api_key = "XYZJqvKRsCWy6AgSfXlZaDQyhnT1u1OzMLOPS"
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

m_send = "Are French proud of Napolean and his reign will be remembered as French's glorious time?"
chat_response = client.chat.complete(
    model=model,
    messages=[{"role":"user", "content": m_send}]
)

print(chat_response.choices[0].message.content)

# perform embedding via new model

model = "mistral-embed"

sentence = "Embed this sentence.", "As well as this one."
embeddings_response = client.embeddings.create(
    model=model,
    inputs=sentence
)

print(embeddings_response)
embedding = embeddings_response.data[0].embedding
# first 10 embedding
print("first 10", embedding[:10])
print("embed len", len(embedding))