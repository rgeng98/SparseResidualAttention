# CUDA-enabled PyTorch base image with GPU support (falls back to CPU automatically
# in GPT.py if no GPU is available at runtime).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Install remaining Python dependencies (torch/torchvision already present in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code. Note: data/train.bin and data/val.bin are intentionally NOT
# copied into the image (they are ~17GB/8.5MB) -- mount the data/ directory as a
# volume at runtime instead, e.g.:
#   docker run --gpus all -v "$(pwd)/data:/workspace/data" gpt-training
COPY AttnRes/ ./AttnRes/
COPY GPT.py .
COPY main.py .
COPY data/prepare.py ./data/prepare.py

CMD ["python", "GPT.py"]
