import os
from pathlib import Path

import onnxruntime.quantization as onnx_quant
import transformers as tfm
from psutil import cpu_count

from . import onnx


def convert_to_ort_and_quantize(
    model_input_path: str, model_output_dir: str, tokenizer: tfm.AutoTokenizer
):
    os.environ["OMP_NUM_THREADS"] = f"{cpu_count()}"
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

    onnx_model_path = Path(os.path.join(model_output_dir, "model.onnx"))
    onnx_quantized_model_path = Path(os.path.join(model_output_dir, "model.quant.onnx"))
    tfm.convert_graph_to_onnx.convert(
        framework="pt",
        model=model_input_path,
        tokenizer=tokenizer,
        output=onnx_model_path,
        opset=12,
        pipeline_name="text-classification",
    )

    onnx_quant.quantize_dynamic(
        onnx_model_path,
        onnx_quantized_model_path,
        weight_type=onnx_quant.QuantType.QInt8,
    )

    return {"quantized_model_path": onnx_quantized_model_path}
