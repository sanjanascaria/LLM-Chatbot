from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma



DATA_PATH = "data"
CHROMA_PATH = "chroma"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

documents = load_documents()
# print(documents[0])

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_id = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_id += 1
        else:
            current_chunk_id = 0 
            last_page_id = current_page_id
        
        chunk_id = f"{current_page_id}:{current_chunk_id}"
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing document chunks in vector database: {len(existing_ids)}")

    new_chunks=[]
    for chunk in chunks_with_ids:
        if chunk.metadata.get("id") not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}.")
        new_chunk_ids = [chunk.metadata.get("id") for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new document chunks to add.")

chunks = split_documents(documents)
add_to_chroma(chunks)






