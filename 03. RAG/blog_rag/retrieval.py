import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

load_dotenv(override=True)
print("Initializing components...")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSION)
llm = ChatOpenAI()
vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}.
    Question: {question}
    Provide a detailed answer. If you don't know the answer, say so.
"""
)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """Simple retrieval chain without using LCEL.
    Manually retrieves documents, formats them and generates a response."""

    # 1. get relevant docs and context
    docs = retriever.invoke(query)
    context = format_docs(docs)

    # 2. Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # 3. Invoke LLM
    response = llm.invoke(messages)
    return response.content


def retrieval_chain_with_lcel():
    """Returns a langchain retrieval chain with LCEL (Runnable).
    This chain can be invoked with {"question":"..."}"""

    # If I dont do anything: format_docs in the chain gets converted as RunnableLambda(format_docs)
    # RunnablePassthrough.assign creates a new dictionary that combines the original input (question) with the new computed field.

    fetcher = itemgetter("question")

    retrieval_chain = (
        RunnablePassthrough.assign(
            context= fetcher | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__=="__main__":
    query = "What is Pinecone in machine learning?"
    print("Retrieving...\n")
    result_without_lcel = retrieval_chain_without_lcel(query=query)
    print("Results without LCEL: \n")
    print(result_without_lcel)
    print("*"*70)

    print("Results with LCEL: \n")
    chain = retrieval_chain_with_lcel()
    response = chain.invoke({"question": query})
    print(response)

