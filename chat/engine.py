from llama_index.core.chat_engine import CondenseQuestionChatEngine

def create_chat_engine(query_engine):
    return CondenseQuestionChatEngine.from_defaults(query_engine=query_engine)
