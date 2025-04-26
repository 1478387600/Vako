import torch
print(torch.cuda.is_available())
print(torch.version.cuda)  # 打印当前的 CUDA 版本

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    # 获取当前 GPU 的名称
    device_name = torch.cuda.get_device_name(0)
    # 检查设备是否支持 FP16
    supports_fp16 = torch.cuda.get_device_capability(0)[0] >= 7  # Volta (7.0) 或更高版本支持 FP16

    print(f"GPU: {device_name}")
    print(f"Supports FP16: {supports_fp16}")
else:
    print("No CUDA device available")