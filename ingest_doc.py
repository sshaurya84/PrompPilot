from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings

import os

# !pip install llama-cpp-python

# !pip install llama-cpp-python

from llama_cpp import Llama

model_path = "C:/Users/sshau/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.1-GGUF/snapshots/731a9fc8f06f5f5e2db8a0cf9d256197eb6e05d1/mistral-7b-instruct-v0.1.Q2_K.gguf"

llm = LlamaCPP(
    model_path=model_path,
    max_new_tokens=512,
    temperature=0.7,
)
# Step 1: Load your PDF
pdf_path = os.path.join("docs", "example.pdf")
documents = PDFReader().load_data(file=pdf_path)


# Step 2: Set the embedding model globally
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = llm
# Step 3: Build vector index
index = VectorStoreIndex.from_documents(documents)

# Step 4: Semantic query
query_engine = index.as_query_engine(llm=None)

results = query_engine.query("What is the return policy?")
print(results)
