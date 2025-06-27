from llama_index.core import Settings
from config.constants import MODEL_PATH, PDF_PATH
from llm.model_loader import load_finetuned_llm
from llm.embedding import load_embedding_model
from index.vector_index import load_and_index_documents
from chat.engine import create_chat_engine
from llama_index.llms.llama_cpp import LlamaCPP

def main():
    fine_tuned_llm = load_finetuned_llm()
    embed_model = load_embedding_model()

    llm = LlamaCPP(
    model_path=MODEL_PATH,
    max_new_tokens=512,
    temperature=0.7,
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

    index = load_and_index_documents(PDF_PATH)
    query_engine = index.as_query_engine(llm)
    chat_engine = create_chat_engine(query_engine)

    print(" PromptPilot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chat_engine.chat(user_input)
        print(f"PromptPilot: {response.response}")

if __name__ == "__main__":
    main()
