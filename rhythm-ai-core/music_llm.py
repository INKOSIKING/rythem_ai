import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RhythmAIMusicLLM:
    def __init__(self, model_name="meta/musicgen-melody"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).eval()

    def generate(self, prompt: str, max_length=256):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)