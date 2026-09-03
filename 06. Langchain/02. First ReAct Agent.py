from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv(override=True)

tavily = TavilySearch()

'''
@tool
def search(query: str) -> str:
    """
    Tool that searches over internet.
    Args:
        query: The query to search for.
    Returns:
        The search result.
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)
'''

llm = ChatOpenAI()
# tools = [search]
tools = [tavily]
agent = create_agent(model=llm, tools=tools)

def main():
    QUERY = "search for 3 job postings for a BI Senior Manager with profile including AI tools and techniques, in Bengaluru, on LinkedIn."
    result = agent.invoke({"messages":HumanMessage(content=QUERY)})
    print(result)

if __name__=='__main__':
    main()