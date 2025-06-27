from llama_index.core import VectorStoreIndex
from llama_index.readers.file import PDFReader
import os

def load_and_index_documents(doc_dir="docs"):

    docs = PDFReader().load_data(file=doc_dir)
    index = VectorStoreIndex.from_documents(docs)
    return index