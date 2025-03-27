# tested by Ram Dhakne
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

api_key = "XYZJqvKRsCWy6AgSfXlZaDQyhnT1u1OzMLOPS"
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)
text = """Large language models (LLMs) have revolutionized natural language processing (NLP).
They enable tasks like text generation, translation, and question answering with high accuracy.
Mistral is one such model known for its efficient processing and capabilities."""

response = client.chat(
    model="mistral-large-latest",
    messages=[ChatMessage(role="user", content=f"Summarize this: {text}")]
)
print(response.choices[0].message.content)
