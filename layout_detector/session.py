"""
Session management for ONNX Runtime inference.
"""

import numpy as np
import onnxruntime
import yaml
from typing import Dict, Any


def create_session(model_path: str, class_mapping_path: str, device: str = "CPU") -> Dict[str, Any]:
    """
    Initialize an ONNX Runtime inference session.

    Args:
        model_path: Path to the .onnx weights file.
        class_mapping_path: Path to metadata.yaml with class names.
        device: "CPU" or "CUDA".

    Returns:
        Dict containing the session object, I/O names, input dimensions,
        class list, and a random color palette for visualization.
    """
    opt_session = onnxruntime.SessionOptions()
    opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = ["CPUExecutionProvider"]
    if device.upper() != "CPU":
        providers.insert(0, "CUDAExecutionProvider")

    session = onnxruntime.InferenceSession(model_path, sess_options=opt_session, providers=providers)

    model_inputs = session.get_inputs()
    input_names = [inp.name for inp in model_inputs]
    input_shape = model_inputs[0].shape
    input_height, input_width = input_shape[2], input_shape[3]

    model_outputs = session.get_outputs()
    output_names = [out.name for out in model_outputs]

    with open(class_mapping_path, "r") as f:
        yaml_data = yaml.safe_load(f)
        # Support both dict-style {0: name} and list-style names
        raw = yaml_data["names"]
        if isinstance(raw, dict):
            classes = [raw[i] for i in sorted(raw.keys())]
        else:
            classes = raw

    color_palette = np.random.uniform(0, 255, size=(len(classes), 3))

    return {
        "session": session,
        "input_names": input_names,
        "input_shape": input_shape,
        "output_names": output_names,
        "input_height": input_height,
        "input_width": input_width,
        "classes": classes,
        "color_palette": color_palette,
    }
