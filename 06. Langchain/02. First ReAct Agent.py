from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from typing import List
from pydantic import BaseModel, Field

load_dotenv(override=True)

tavily = TavilySearch()

class Source(BaseModel):
    """schema for a source used by the agent"""
    url: str = Field(description="URL of the job posting")

class AgentResponse(BaseModel):
    """Schema for the agent response"""
    answer: str = Field(description="Agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, 
                                  description="List of sources used to generate the answer")

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

llm = ChatOpenAI(model="gpt-4o-mini")
# tools = [search]
tools = [tavily]
agent = create_agent(model=llm, 
                     tools=tools,
                     response_format=AgentResponse)

def main():
    QUERY = "search for 3 job postings for a BI Senior Manager with profile including AI tools and techniques, in Bengaluru, on LinkedIn."
    result = agent.invoke({"messages":HumanMessage(content=QUERY)})
    print(result)

if __name__=='__main__':
    main()