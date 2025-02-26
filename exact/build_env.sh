ml --force purge

ml Stages/2025 GCCcore/.13.3.0 Python/3.12.3 CUDA/12 PyTorch/2.5.1

python3 -m venv nano_gpt_env_juwels

source nano_gpt_env_juwels/bin/activate
  
pip3 install ray numpy transformers datasets tiktoken wandb tqdm matplotlib