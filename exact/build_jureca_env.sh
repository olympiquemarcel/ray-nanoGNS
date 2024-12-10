ml Python CUDA NCCL

python3 -m venv nano_gpt_env

source nano_gpt_env/bin/activate
  
pip3 install --no-cache-dir torch==2.1.0+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

pip3 install ray numpy transformers datasets tiktoken wandb tqdm