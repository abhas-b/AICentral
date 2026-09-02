from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")


url = "https://en.wikipedia.org/wiki/Elon_Musk"
headers = {
    "User-Agent": "Mozilla/5.0"
}

def get_content(url):
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Wikipedia article body
    content = soup.select_one("#mw-content-text")

    # Extract paragraphs
    paragraphs = content.find_all("p")

    for p in paragraphs:
        text = p.get_text(" ", strip=True)
        if text:
            return text

def main():
    text = get_content(url=url)

    summary_template = """
    given the information {information} about a person, i want you to create:
    1. Short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"],
                                             template=summary_template)

    llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')

    chain = summary_prompt_template | llm
    response = chain.invoke(input={'information':text})
    print(response.content)


if __name__ == "__main__":
    main()