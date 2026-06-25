import torch

print("--- GPU System Check ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs found: {gpu_count}")
    print(f"Active GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: PyTorch cannot see your GPU. It is defaulting to CPU.")