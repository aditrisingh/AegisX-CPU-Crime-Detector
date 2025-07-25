import torch
from mc3 import get_mc3_model
import os

# Step 1: Load model
model = get_mc3_model()
model.eval()  # VERY IMPORTANT

# Step 2: Dummy input - [batch, channel, frames, height, width]
dummy_input = torch.randn(1, 3, 16, 112, 112)

# Step 3: Export
onnx_file_path = "mc3_model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'frames'}, 
        'output': {0: 'batch_size'}
    }
)

print(f"✅ Exported MC3 model to {onnx_file_path}")
