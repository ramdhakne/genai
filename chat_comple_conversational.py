# tested by Ram Dhakne
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

api_key = "XYZJqvKRsCWy6AgSfXlZaDQyhnT1u1OzMLOPS"
client = MistralClient(api_key=api_key)

messages = [
    ChatMessage(role="system", content="You are an AI assistant."),
    ChatMessage(role="user", content="What is the capital of Japan?")
]

response = client.chat(model="mistral-large-latest", messages=messages)
print(response.choices[0].message.content)
