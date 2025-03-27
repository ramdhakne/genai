# tested by Ram Dhakne
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

API_KEY = "XYZJqvKRsCWy6AgSfXlZaDQyhnT1u1OzMLOPS"

client = MistralClient(api_key=API_KEY)

response = client.chat(
    model="mistral-large-latest",
    messages=[ChatMessage(role="user", content="Write a Python function to check if a number is prime.")]
)
print(response.choices[0].message.content)