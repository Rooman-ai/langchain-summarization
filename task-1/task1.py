import os
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
api_version = os.getenv("API_VERSION")


print("Azure OpenAI API Key:", api_key)
print("Endpoint URL:", endpoint)
print("Deployment Name:", deployment)
print("API Version:", api_version)
