# export_mc3_features.py

import torch
from last_feature_extractor import get_mc3_feature_extractor

model = get_mc3_feature_extractor("cpu")  # or "cuda"
dummy_input = torch.randn(1, 3, 16, 112, 112)  # (B, C, T, H, W)

torch.onnx.export(
    model,
    dummy_input,
    "mc3_features.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 2: "frames"}},
    opset_version=12
)

print("✅ Exported MC3-18 feature extractor to mc3_features.onnx")
