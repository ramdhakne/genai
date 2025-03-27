# tested by Ram Dhakne
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# works only on pip install mistralai==0.4.2 

API_KEY = "XYZJqvKRsCWy6AgSfXlZaDQyhnT1u1OzMLOPS"

client = MistralClient(api_key=API_KEY)

def sentiment_analysis():

    sentence = "The product is amazing, but the delivery was too slow."

    response = client.chat(
        model="mistral-large-latest",
        messages=[ChatMessage(role="user", content=f"Analyze the sentiment of this sentence: {sentence}")]
    )
    print(response.choices[0].message.content)

# NER - named entiry recognition
def ner():
    sentence = "Apple Inc. is headquartered in Cupertino, California, and was founded by Steve Jobs."

    response = client.chat(
        model="mistral-large-latest",
        messages=[ChatMessage(role="user", content=f"Extract named entities from: {sentence}")]
    )
    print(response.choices[0].message.content)


def code_gen():
    response = client.chat(
        model="mistral-large-latest",
        messages=[ChatMessage(role="user", content="Write a Python function to check if a number is even.")]
    )
    print(response.choices[0].message.content)


# Run all examples
def main():
    code_gen()
    ner()
    sentiment_analysis()

if __name__ == "__main__":
    main()