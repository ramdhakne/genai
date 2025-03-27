# tested by Ram Dhakne
from mistralai.client import MistralClient

API_KEY = "XYZJqvKRsCWy6AgSfXlZaDQyhnT1u1OzMLOPS"

client = MistralClient(api_key=API_KEY)

models = client.list_models()
print(models)