from get_embedding_function import get_embedding_function
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import argparse

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query: str):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
        )
    
    results = db.similarity_search_with_score(query=query, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    print(prompt)

    model = Ollama(model="mistral-nemo:latest")

    response = model.invoke(prompt)
    source = [doc.metadata.get("id") for doc, _ in results]
    final_response = f"{response}\nSources: {source}"

    print(final_response)
    return final_response

if __name__ == "__main__":
    main()

