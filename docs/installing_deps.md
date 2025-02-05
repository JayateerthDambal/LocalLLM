## Installing Dependencies for running LLM Model

---

1. First install PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- I am using Cuda V11.8, as this is most stable one. If you are using different version, you might need to install different version of PyTorch. Look for their documentation
- If you are using CPU, you can install CPU version using simpe PyPi install command.

- Check if you have CUDA installed by running `nvcc --version` in your terminal. If you don't have CUDA you can browse on the official NVIDIA website to download and install it. I prefer CUDA 11.8 as it is most stable one.

