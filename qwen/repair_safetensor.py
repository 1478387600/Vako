import torch
from safetensors.torch import load_file, save_file

weights = load_file('qwen/model.safetensors')

# 检查并修复
for name, tensor in weights.items():
    if torch.isnan(tensor).any():
        print(f"Found NaNs in {name}, fixing...")
        tensor = torch.nan_to_num(tensor, nan=0.0)  # 把NaN换成0
        weights[name] = tensor

# 保存修复后的文件
save_file(weights, './model.safetensors')
