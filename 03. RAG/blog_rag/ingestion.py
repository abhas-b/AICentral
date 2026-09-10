from dotenv import load_dotenv
import os
# from langchain_community.document_loaders import TextLoader -- now deprecated

from pathlib import Path
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = BASE_DIR / "blog_text.txt"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# num_tokens_from_string("tiktoken is great!", "cl100k_base")

if __name__=="__main__":
    
    if not KNOWLEDGE_BASE.exists():
        raise FileNotFoundError(f"File not found: {KNOWLEDGE_BASE}")

    with KNOWLEDGE_BASE.open("r", encoding="utf-8") as f:
        file = f.read()
        # print(f"Number of tokens to be ingested... {num_tokens_from_string(string=file, encoding_name="cl100k_base")}")
    print("Ingesting...")

    # Load
    loader = UnstructuredLoader(file_path=KNOWLEDGE_BASE,
                                chunking_strategy="basic",
                                max_characters=1000000,
                                encoding="UTF-8")
    document = loader.load()
    print(f"Number of documents loaded: {len(document)}")

    # Split
    print("Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
    chunks = text_splitter.split_documents(documents=document)
    print(f"Splitted text into {len(chunks)} chunks")

    # Embedding
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSION)

    print("Ingesting the chunks into VectorDB...")
    PineconeVectorStore.from_documents(chunks,
                                       embedding=embeddings,
                                       index_name=os.environ['INDEX_NAME'])