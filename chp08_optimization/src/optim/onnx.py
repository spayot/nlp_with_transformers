# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
from scipy.special import softmax


def create_model_for_provider(
    model_path: Path, provider: str = "CPUExecutionProvider"
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), options, providers=[provider])
    session.disable_fallback()

    return session


class OnnxPipeline:
    def __init__(self, model, tokenizer, id2labels: str):
        self.model = model
        self.tokenizer = tokenizer
        self.id2labels = id2labels

    def __call__(self, query: list[str]) -> list[dict]:
        model_inputs = self.tokenizer(query, return_tensors="pt")
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()

        return [{"label": self.id2labels(pred_idx), "score": probs[pred_idx]}]
