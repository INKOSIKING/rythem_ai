import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx(model_path, quantized_model_path):
    quantize_dynamic(
        model_path,
        quantized_model_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized ONNX model saved to {quantized_model_path}")

if __name__ == "__main__":
    quantize_onnx("models/checkpoints/melodygen.onnx", "models/checkpoints/melodygen.quant.onnx")