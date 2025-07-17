import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
session = ort.InferenceSession("models/checkpoints/melodygen.quant.onnx")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    input_ids = np.array(data["input_ids"], dtype=np.int64)
    ort_inputs = {"input": input_ids}
    ort_outs = session.run(None, ort_inputs)
    return jsonify({"output": ort_outs[0].tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)