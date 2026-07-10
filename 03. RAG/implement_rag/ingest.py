import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import warnings
import glob

warnings.filterwarnings("ignore")


DB_NAME = str(Path(__file__).parent.parent/"vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent/"knowledge-base")
embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
load_dotenv(override=True)


def fetch_documents():
    documents = []
    folders = glob.glob(str(Path(KNOWLEDGE_BASE)/"*"))
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, 
                            glob="**/*.md",
                            loader_cls=TextLoader, 
                            loader_kwargs={'encoding':'utf-8'}, 
                            show_progress=True)
        docs = loader.load()
        for doc in docs:
            doc.metadata['doc_type'] = doc_type
    
        documents.extend(docs)
    return documents

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(embedding_function=embeddings, persist_directory=DB_NAME).delete_collection()
    
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_NAME)
    print(f"Created {vectorstore._collection.count()} Vectors with {len(vectorstore._collection.get(limit=1, include=['embeddings'])['embeddings'][0])} dimensions")
    return vectorstore


if __name__=='__main__':
    print(DB_NAME)
    print(KNOWLEDGE_BASE)
    documents = fetch_documents()
    chunks = create_chunks(documents)
    vectorstore = create_embeddings(chunks)
    print("Ingestion Complete!")