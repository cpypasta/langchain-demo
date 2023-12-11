import os, json, requests, streamlit as st
from dotenv import load_dotenv, find_dotenv
from typing import Type
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType, tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.schema import SystemMessage
from langchain.schema.agent import AgentFinish
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.globals import set_debug, set_verbose

# Setup
load_dotenv(find_dotenv())
google = GoogleSerperAPIWrapper()
gpt3_5 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=True)
set_debug(False)
set_verbose(True)

# Creates tools or references to existing tools
@tool
def get_word_length(word: str) -> int:
  """Returns the length of a word."""
  return len(word)

agent_tools = [
  Tool(
    name="Search",
    func=google.run,
    description="useful for when you need information about current events and data"
  ),
  get_word_length
]


# Creates Prompts
# MessagesPlaceholder is a placeholder for a list of messages. `agent_scratchpad` is specifically used 
# to store the descriptions of the tools available.
prompt = ChatPromptTemplate.from_messages([
  (
    "system",
    "You are a powerful assistant, but you are bad at calculating lengths of words."
  ),
  ("user", "{input}"),
  MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Have the LLM  aware of the tools available; in this case for the OpenAI API
openai_tools = [format_tool_to_openai_function(t) for t in agent_tools]
print(json.dumps(openai_tools, indent=2))
gpt3_5_with_tools = gpt3_5.bind(functions=openai_tools)


agent = (
  {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"])
  }
  | prompt | gpt3_5_with_tools | OpenAIFunctionsAgentOutputParser()
)

user_input = "get the current date using python"


def manual_invoke():
  intermediate_steps = []
  while True:
    output = agent.invoke({
      "input": user_input,
      "intermediate_steps": intermediate_steps
    })
    if isinstance(output, AgentFinish):
      final_result = output.return_values["output"]
      break
    else:
      print(f"TOOL NAME: {output.tool}")
      print(f"TOOL INPUT: {output.tool_input}")
      tool = {"get_word_length": get_word_length}[output.tool]
      observation = tool.run(output.tool_input)
      intermediate_steps.append((output, observation))
      
  return final_result
    
    
# print(manual_invoke())

agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True)
agent_executor.invoke({"input": user_input})

### ADD MEMORY
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages([
  (
    "system",
    "You are a powerful assistant, but you are bad at calculating lengths of words."
  ),
  MessagesPlaceholder(variable_name=MEMORY_KEY),
  ("user", "{input}"),
  MessagesPlaceholder(variable_name="agent_scratchpad")
])
chat_history = []
agent = (
  {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
    "chat_history": lambda x: x[MEMORY_KEY]
  }
  | prompt | gpt3_5_with_tools | OpenAIFunctionsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True)

# result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
# chat_history.extend([
#   HumanMessage(content=user_input),
#   AIMessage(content=result["output"])
# ])
# agent_executor.invoke({"input": "is that a real word?", "chat_history": chat_history})