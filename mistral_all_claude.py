# tested by Ram Dhakne
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Initialize the Mistral client
api_key = "XYZJqvKRsCWy6AgSfXlZaDQyhnT1u1OzMLOPS"
#api_key = os.environ.get('MISTRAL_API_KEY')
client = MistralClient(api_key=api_key)

# Example 1: Basic Chat Completion
def basic_chat_completion():
    messages = [
        ChatMessage(role="user", content="Explain quantum computing in simple terms")
    ]
    
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages
    )
    print("Basic Chat Response:", chat_response.choices[0].message.content)

# Example 2: Conversation with Multiple Messages
def multi_turn_conversation():
    messages = [
        ChatMessage(role="user", content="I want to plan a trip to Japan"),
        ChatMessage(role="assistant", content="Great! What kind of experience are you looking for?"),
        ChatMessage(role="user", content="I'm interested in traditional culture and modern cities")
    ]
    
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages
    )
    print("Multi-turn Conversation Response:", chat_response.choices[0].message.content)

# Example 3: Code Generation
def code_generation():
    messages = [
        ChatMessage(role="user", content="Write a Python function to implement a binary search algorithm")
    ]
    
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages
    )
    print("Code Generation Response:", chat_response.choices[0].message.content)

# Example 4: JSON Output Structuring
def structured_output():
    messages = [
        ChatMessage(role="user", content="Extract key information from this text as a JSON. Provide name, age, and occupation")
    ]
    
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages,
        response_format={"type": "json_object"}
    )
    print("Structured Output:", chat_response.choices[0].message.content)

# Example 5: Temperature and Top-P Sampling
def creative_writing():
    messages = [
        ChatMessage(role="user", content="Write a short story about a robot discovering emotions")
    ]
    
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.7,  # Increases randomness/creativity
        top_p=0.9         # Controls diversity of token selection
    )
    print("Creative Writing Response:", chat_response.choices[0].message.content)

# Example 6: Language Translation
def translation_example():
    messages = [
        ChatMessage(role="user", content="Translate the following English text to French: 'Hello, how are you doing today?'")
    ]
    
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages
    )
    print("Translation Response:", chat_response.choices[0].message.content)

# Example 7: System + User Message Interaction
def system_message_example():
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant that speaks like a pirate"),
        ChatMessage(role="user", content="Can you help me understand blockchain technology?")
    ]
    
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages
    )
    print("Pirate Assistant Response:", chat_response.choices[0].message.content)

# Run all examples
def main():
    basic_chat_completion()
    multi_turn_conversation()
    code_generation()
    structured_output()
    creative_writing()
    translation_example()
    system_message_example()

if __name__ == "__main__":
    main()