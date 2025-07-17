from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def lora_finetune(base_model_name, train_loader, config):
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = get_peft_model(model, lora_cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    for epoch in range(config["training"]["epochs"]):
        for batch in train_loader:
            input_ids = batch["input"]
            labels = batch["target"]
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print("LoRA fine-tuning complete.")
    return model