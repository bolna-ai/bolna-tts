
import requests
import json
from base64 import b64decode
url = "http://127.0.0.1:4680/tts"

payload = json.dumps({
  "model": "StyleTTS",
  "config": {
    "text": "Hello World I am R Ansh Joseph",
    "sr": 8000
  }
})
headers = {
  'accept': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
response = json.loads(response.text)

file = b64decode(response['audio'])
with open("test.wav",'wb') as f:
    f.write(file)
