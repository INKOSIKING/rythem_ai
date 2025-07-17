import requests

class RythemAISDK:
    def __init__(self, api_key, base_url="https://api.rythemai.com"):
        self.api_key = api_key
        self.base_url = base_url

    def chat(self, message, user_id="user1"):
        r = requests.post(f"{self.base_url}/chat/", json={
            "user_id": user_id,
            "message": message
        }, headers={"Authorization": f"Bearer {self.api_key}"})
        return r.json()

    def get_music(self, music_id):
        r = requests.get(f"{self.base_url}/music/{music_id}", headers={"Authorization": f"Bearer {self.api_key}"})
        return r.content