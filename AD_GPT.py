# Import necessary libraries
import re  # For regular expression matching
from typing import List, Union  # Type hints for clarity
import textwrap  # For formatting long text output
import time  # To create delays for a typing animation effect
import os  # For file and folder operations
from duckduckgo_search import ddg  # DuckDuckGo search API wrapper
from AD_GPT_tools import (  # Custom tools for scraping and analysis
    scrape_text, scrape_links, scrape_place_text, 
    get_summary_period, text_all_lda, get_city_info
)
import descartes  # Used for geographic plotting (optional)
from dateparser.search import search_dates  # To extract dates from text
from requests.packages import urllib3  # Suppress SSL warnings

# LangChain libraries for building and managing LLM-based agents
from langchain.agents import (
    Tool, AgentExecutor, LLMSingleActionAgent, 
    initialize_agent, AgentOutputParser
)
from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain import OpenAI, LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import AgentType
import shutil  # For copying files between directories

# Configure the text splitter to handle large documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,  # Size of each chunk
    chunk_overlap=0  # No overlap between chunks
)

# Template for context-based question answering
CONTEXT_QA_TMPL = """
Answer user's questions according to the information provided below
Information: {context}

Question: {query}
"""

# PromptTemplate object for context-based Q&A
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# Function to simulate a typing effect when printing responses
def output_response(response: str) -> None:
    if not response:
        exit(0)  # Exit if the response is empty
    # Print each character with a delay to simulate typing
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # 0.1-second delay per character
            print(" ", end="", flush=True)  # Add space between words
        print()  # Move to the next line
    print("----------------------------------------------------------------")

# Class to manage AD-GPT functions such as search, summary, and visualization
class ADGPT:
    def __init__(self, llm):
        """Initialize with a language model."""
        self.llm = llm

    def google_search(self, query: str) -> str:
        """Search news using DuckDuckGo and save results locally."""
        urls = []
        pwd = os.getcwd()  # Get current working directory
        # Create the 'workplace' folder if it doesn't exist
        if not os.path.isdir(pwd + "/workplace"):
            os.mkdir(pwd + "/workplace")
        # Check if news links are already saved
        if os.path.exists(pwd + "/workplace/bbc_news_links.txt"):
            print("News has already been saved on this device")
        # Perform a DuckDuckGo search
        re = ddg(query, max_results=20)
        if re is None:
            # Fallback in case of network issues
            news_links = os.listdir(pwd + '\\news_happend_lastyear\\')
            for news_link in news_links:
                shutil.copyfile(
                    pwd + '\\news_happend_lastyear\\' + news_link,
                    pwd + '\\workplace\\' + news_link
                )
            return "Internet error, but stored news links are available."
        # Save search results to a file
        with open(pwd + "/workplace/news_links.txt", "w", encoding='utf8') as file:
            for web in re:
                url = web['href']
                print(url)
                file.write(url + "\n")
                urls.append(url)
        return "The latest news has been saved on this device."

    def draw_news(self, query: str) -> str:
        """Generate summaries and visualizations from saved news data."""
        save_dir = os.getcwd() + '/workplace/news_summary/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # Generate topic and place summaries
        text_result = get_summary_period(save_dir, save_dir)
        text_all_lda(text_result, save_dir)
        get_city_info(save_dir)
        return "Everything you need has been obtained."

    def summary_news(self, query: str) -> str:
        """Summarize saved news articles and extract relevant details."""
        pwd = os.getcwd()
        Dir = pwd + "/workplace/"
        news_web_list = ['CNN', 'Fox', 'Hill', 'NPR', 'USAToday']

        # Process each news website's saved articles
        for web in news_web_list:
            urls = []
            if os.path.exists(Dir + web + '_news_links.txt'):
                with open(Dir + web + '_news_links.txt', 'r', encoding='utf8') as f:
                    urls = [url.strip() for url in f.readlines()]

            # Extract information from each URL
            for count, url in enumerate(urls, 1):
                print(f'Browsing {url} and saving information...')
                web_dir = f"{Dir}{web}/news_{count}/"
                os.makedirs(web_dir, exist_ok=True)

                # Scrape news text, dates, and titles
                text, datetime, news_title = scrape_text(url, web)
                with open(web_dir + "text.txt", "w", encoding='utf8') as f:
                    f.write(text)

                # Extract and save places mentioned
                cities = scrape_place_text(text)
                with open(web_dir + "places.txt", "w", encoding='utf8') as f:
                    f.writelines(city + '\n' for city in cities)

                # Create and save a summary of the news article
                docs = [Document(page_content=text)]
                summary = load_summarize_chain(self.llm, "map_reduce").run(docs)
                with open(web_dir + "summary.txt", "w", encoding='utf8') as f:
                    f.write(summary)
        return "News summary saved in the workplace folder."

    def introduce_info(self, query: str) -> str:
        """Introduce the AD-GPT system."""
        context = """
        AD-GPT is an automated system that collects, summarizes, and organizes 
        information on Alzheimer's disease using powerful reasoning from large language models.
        """
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        return self.llm(prompt)

# Custom template for formatting agent prompts
class CustomPromptTemplate(StringPromptTemplate):
    template: str  # The base template
    tools: List[Tool]  # List of available tools

    def format(self, **kwargs) -> str:
        """Format the template with given variables."""
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Custom output parser to interpret LLM outputs
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """Parse the output of the LLM."""
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        match = re.search(r"Action\s*:(.*?)\nAction Input\s*:(.*)", llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        return AgentAction(
            tool=match.group(1).strip(),
            tool_input=match.group(2).strip(),
            log=llm_output,
        )

# Main execution logic
if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") ## This can be changed to gpt-4, gpt-4o, etc.
    urllib3.disable_warnings()
    ad_gpt = ADGPT(llm)
    ## Define the basic functions of AD-AutoGPT
    tools = [
        Tool("Search for news", ad_gpt.google_search, "Search latest AD news."),
        Tool("Summarize news", ad_gpt.summary_news, "Summarize saved AD news."),
        Tool("Introduce AD-AutoGPT", ad_gpt.introduce_info, "Introduce the system."),
        Tool("Draw plots", ad_gpt.draw_news, "Visualize news topics and trends."),
    ]

    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL, tools=tools, input_variables=["input", "intermediate_steps"]
    )
    output_parser = CustomOutputParser()

    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)
    ## AD-AutoGPT will keep running until reaches the final goal.
    while True:
        try:
            user_input = input("Please enter your question: ")
            response = agent_executor.run(user_input)
            output_response(response)
        except KeyboardInterrupt:
            break
