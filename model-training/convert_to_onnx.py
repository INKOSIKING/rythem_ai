import torch

def convert(model, dummy_input, out_path):
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17
    )

if __name__ == "__main__":
    # Example for MelodyGenModel
    from models.melodygen import MelodyGenModel
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    model = MelodyGenModel(cfg["model"])
    model.load_state_dict(torch.load("models/checkpoints/best_model.pt"))
    model.eval()
    dummy_input = torch.randint(0, cfg["model"].get("vocab_size", 512), (1, 128))
    convert(model, dummy_input, "models/checkpoints/melodygen.onnx")
    print("Exported ONNX model.")