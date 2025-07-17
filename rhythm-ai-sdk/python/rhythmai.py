import requests

class RhythmAI:
    def __init__(self, api_key, base_url="https://api.rhythmai.com"):
        self.api_key = api_key
        self.base_url = base_url

    def chat(self, prompt):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(f"{self.base_url}/chat/", json={"prompt": prompt}, headers=headers)
        resp.raise_for_status()
        return resp.json()