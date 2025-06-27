import os

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.1-GGUF/snapshots/731a9fc8f06f5f5e2db8a0cf9d256197eb6e05d1/mistral-7b-instruct-v0.1.Q2_K.gguf")
HF_MODEL_ID = "mistralai/Mistral-7B-v0.1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PDF_PATH = os.path.join("docs", "example.pdf")
TEMPERATURE = 0.7
TRAINING_MODEL = "EleutherAI/gpt-neo-1.3B"
