import re
from typing import List, Union
import textwrap
import time
import os 
from duckduckgo_search import ddg
from AD_GPT_tools import scrape_text, scrape_links, scrape_place_text
import descartes
from dateparser.search import search_dates
from requests.packages import urllib3
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate

from langchain.llms.base import BaseLLM
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

text_splitter = CharacterTextSplitter()

CONTEXT_QA_TMPL = """
Answer user's questions according to the information provided below
Information：{context}

Question：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)


def output_response(response: str) -> None:
    if not response:
        exit(0)
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # Add a delay of 0.1 seconds between each character
            print(" ", end="", flush=True)  # Add a space between each word
        print()  # Move to the next line after each line is printed
    print("----------------------------------------------------------------")


class ADGPT:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
    
    def google_search(self, query: str) -> None:
        
        urls= []
        pwd = os.getcwd()
        if not os.path.isdir(pwd+"/workplace"):
            os.mkdir(pwd+"/workplace")

        re = ddg(query, max_results=20)
        if(len(re)==0):
            return
        with open(pwd+"/workplace/news_links.txt", "w", encoding='utf8') as file:
            for web in re:
                url = web['href']
                print(url)
                file.write(url + "\n")
                urls.append(url) 
                
    def analyze_news(self, query: str) -> None:
        pwd = os.getcwd()
        if not os.path.isdir(pwd+"/workplace"):
            os.mkdir(pwd+"/workplace")
        Dir = pwd+"/workplace/"
        urls = []
        news_web_list = ['ARUK']
        for web in news_web_list:
            if os.path.exists(pwd+'/workplace/'+web+'_news_links.txt'):
                with open(pwd+'/workplace/'+web+'_news_links.txt','r',encoding='utf8') as f:
                    for url_tmp in f.readlines():
                        if(url_tmp != '\n'):
                            urls.append(url_tmp)
            else:
                print('There are no news saved on local device')
                return
            count = 0
            # print(urls)
            for url in urls:
                count = count +1 
                print('\nBrowsing'+str(url)+'and save useful infomation in workplace folder...')
                if not os.path.isdir(Dir+web):
                    os.mkdir(Dir+web)
                if not os.path.isdir(Dir+web+"/news_"+str(count)):
                    os.mkdir(Dir+web+"/news_"+str(count))
                Dir1 = Dir+web+"/news_"+str(count)
                text,datetime = scrape_text(url.replace('\n',''),web)
                with open(Dir1 + "/text.txt", "w", encoding='utf8') as f:
                    f.write(text)
                cities = scrape_place_text(text)
                with open(Dir1 + "/places.txt", "w", encoding='utf8') as f:
                    for city in cities:
                        f.writelines(city+'\n')     
                links = scrape_links(url)
                with open(Dir1 + "/links.txt", "w", encoding='utf8') as f:
                    for link in links:
                        f.writelines(link+'\n')  
                with open(Dir1 + "/dates.txt", "w", encoding='utf8') as f:
                    f.writelines(datetime+'\n')  
                    
                with open(Dir1 + "/text.txt", "r", encoding='utf8') as f:
                    state_of_the_union = f.read()
                texts = text_splitter.split_text(state_of_the_union)
                docs = [Document(page_content=t) for t in texts[:3]]
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary_file = chain.run(docs)
                with open(Dir1 + "/summary.txt", "w", encoding='utf8') as f:
                    f.write(summary_file)
                
    def introduce_info(self, query: str) -> str:
        """introduce AD-GPT"""
        context = """
        With the powerful reasoning ability of LLM, we have built an automated task system similar to AutoGPT, \
            called AD-GPT, to collect and organize information related to AD on a daily basis. \
                AD-GPT has search, summary, storage, and drawing capabilities. With our own tools, \
                    it can automatically run relevant tasks and organize them without human intervention.
        """
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        return self.llm(prompt)


AGENT_TMPL = """Answer the following questions in the given format, You can use the following tools：

{tools}

When answering, please follow the format enclosed in ---

---
Question: The question need to be answered
Thought: What should I do to answer the above question
Action: choose one tool from ”{tool_names}“ 
Action Input: choose the input_args that action requires
Observation: Choose the results returned by tools
...（The action of thinking/observation can repeat N times）
Thought: Now, I've got the final answer
Final Answer: The final answer of the initial question
---

Now start to answer user's questions, remember to follow the specified format step by step before providing the final answer.

Question: {input}
{agent_scratchpad}
"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str  # Standard template
    tools: List[Tool]  # Usable tools

    def format(self, **kwargs) -> str:
        """
        Fill in all the necessary values according to the defined template.
        
        Returns:
            str: filled template。
        """
        intermediate_steps = kwargs.pop("intermediate_steps")  # Extract the intermediate steps and execute them.
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts  # Record the thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )  # Enumerate all available tool names and tool descriptions
        kwargs["tool_names"] = ", ".join(
            [tool.name for tool in self.tools]
        )  # Enumerate all tools' names
        cur_prompt = self.template.format(**kwargs)
        print(cur_prompt)
        return cur_prompt


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        Interpret the output of LLM and locate the necessary actions based on the output text.

        Args:
            llm_output (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            Union[AgentAction, AgentFinish]: _description_
        """
        if "Final Answer:" in llm_output:  # If the sentence contains "Final Answer", it means it has been completed.
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"  # Interpret action_input and action
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


if __name__ == "__main__":
    ## set api token in terminal
    
    os.environ["OPENAI_API_KEY"] = "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"
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    urllib3.disable_warnings()
    ad_gpt = ADGPT(llm)
    tools = [
        Tool(
            name="search and save the latest Alzheimer's disease news",
            func=ad_gpt.google_search,
            description="This is a tool that use the Google to search for the latest news about Alzhemier's disease and save the URLs in a file",
        ),
        Tool(
            name="Analyze the news",
            func=ad_gpt.analyze_news,
            description="This is a tool to know when and where the Alzheimer's news happens,which will extract and save the time, place and hyperlinks in the news",
        ),
        Tool(
            name="Introduce AD-GPT",
            func=ad_gpt.introduce_info,
            description="This is a tool to introduce AD-GPT",
        ),
    ]
    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )
    output_parser = CustomOutputParser()

    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    while True:
        try:
            user_input = input("Please enter your question: ")
            response = agent_executor.run(user_input)
            output_response(response)
        except KeyboardInterrupt:
            break