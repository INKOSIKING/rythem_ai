import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from rythemai.models.text_to_music_transformer import TextToMusicTransformer
from rythemai.data.dataset import TextToMusicDataset
from rythemai.utils.logger import setup_logger

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("rythemai/config/config.yaml")
    logger = setup_logger("train", config["logging"]["file"], config["logging"]["level"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = TextToMusicDataset(
        config["data"]["meta_csv"],
        config["data"]["midi_dir"],
        config["data"]["vocab_file"],
        config["data"]["max_seq_len"]
    )
    loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"])

    # Model
    vocab_size = len(dataset.tokenizer.stoi)
    model = TextToMusicTransformer(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.stoi["PAD"])
    best_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for text, midi in loader:
            text, midi = text.to(device), midi.to(device)
            tgt_input = midi[:, :-1]
            tgt_output = midi[:, 1:]
            output = model(text, tgt_input)
            output = output.view(-1, vocab_size)
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * text.size(0)

        epoch_loss = running_loss / len(dataset)
        logger.info(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {epoch_loss:.4f}")

        # Checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config["training"]["checkpoint_dir"], "best_model.pt"))
            logger.info(f"Best model saved at epoch {epoch+1} with loss {epoch_loss:.4f}")

if __name__ == "__main__":
    main()