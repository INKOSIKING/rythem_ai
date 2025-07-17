import os
import openai

class LLMChatWrapper:
    def __init__(self, api_key=None, model="gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        openai.api_key = self.api_key

    def chat(self, prompt, history=None):
        messages = [{"role": "system", "content": "You are a helpful AI music assistant."}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()