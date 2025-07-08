from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model