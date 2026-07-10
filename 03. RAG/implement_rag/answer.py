import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document


DB_NAME = str(Path(__file__).parent.parent/"vector_db")
embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
MODEL = "gpt-4.1-nano"
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
RETRIEVAL_K = 5


def read_vectorstore(db_name, embedding_function):
    vectors = Chroma(
    persist_directory=db_name,
    embedding_function=embedding_function
    )
    return vectors



llm = ChatOpenAI(model_name=MODEL, temperature=0)
retriever = read_vectorstore(DB_NAME, embeddings).as_retriever()


SYSTEM_PROMPT_TEMPLATE = """You are an expert on InsureLM, a company that provides insurance related products.
You are able to answer questions related to InsureLM, its products and employees. 
You are provided additional context that might be relevant to a user's question. 
Give brief, accurate answers.
If you don't know the answer to any question, just say so.
Respond in less than 100 words.

Context:
{context}
"""

def fetch_context(question: str)  -> list[Document]:
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """Combined user question with history into a single string"""
    prior = "\n".join(m['content'] for m in history if m['role'] == 'user')
    return prior + '\n\n' + question

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """Answer the given question with RAG. Return the answer and context documents"""

    # combine history
    combined = combined_question(question, history)

    # get additional context
    docs = fetch_context(question=combined)
    context = "\n\n".join(doc.page_content for doc in docs)

    # print(f"Additional Context: \n\n {context}")
    # Add context to system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
    messages = [SystemMessage(content=system_prompt)]

    for msg in history:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            messages.append(AIMessage(content=msg['content']))
    # messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))


    response = llm.invoke(messages)
    return response.content, docs