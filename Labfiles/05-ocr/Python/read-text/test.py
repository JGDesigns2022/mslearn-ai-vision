import requests
import os

ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')

headers = {
    'Ocp-Apim-Subscription-Key': ai_key,
    'Content-Type': 'application/json'
}

# Test the endpoint with a simple request
url = ai_endpoint + "/vision/v3.2/read/analyze"  # Check if this is the correct endpoint
response = requests.post(url, headers=headers)

print(response.status_code)
print(response.text)
