from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class LyricsGenModel:
    """
    Transformer-based lyric generator using HuggingFace Seq2Seq model (e.g., T5, BART).
    """
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)