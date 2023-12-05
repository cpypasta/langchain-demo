import requests, json, os, time
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from bs4 import BeautifulSoup
from openrouter import OpenRouter

load_dotenv(find_dotenv())
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


def optimize_query(llm, query):
  template = """
  QUERY:
  {query}

  INSTRUCTIONS:
  You are an expert on how to perform Google searches. When receiving a search query, your job is to optimize the query provided above to get the best results. Ensure all acronyms are converted into their words forms and ambiguous language is corrected. Only return the optimized query with no other text. DO NOT return any text that should not be included in the query. DO NOT start the response with "the optimized query is".

  OPTIMIZED QUERY:  
  """
  prompt = PromptTemplate.from_template(template)
  run = prompt | llm | StrOutputParser()
  return run.invoke({ "query": query })


def search(query):
  url = "https://google.serper.dev/search"

  payload = json.dumps({
    "q": query
  })
  headers = {
    'X-API-KEY': SERPAPI_API_KEY,
    'Content-Type': 'application/json'
  }

  start_time = time.time()
  response = requests.request("POST", url, headers=headers, data=payload)
  result = response.json()["organic"]
  result.sort(key=lambda x: int(x["position"]))
  print("--- Search: %s seconds ---" % (time.time() - start_time))
  return result


def find_best_article_urls(llm, response_data, query, top_n=True):  
  if top_n:
    return [x["link"] for x in response_data[0:3]]
  else:
    response_str = json.dumps(response_data)
    template = """
    You are a world class journalist & researcher, you are extremely good at finding the most relevant articles pertaining to a certain topic.
    
    {response_str}
    
    Above is the list of search results for the query "{query}".
    Please choose the best 3 articles from the list. Return ONLY an array of the urls. Do not include anything else.
    """

    prompt_template = PromptTemplate(
      input_variables=["response_str", "query"], template=template
    )
    article_picker_chain = LLMChain(
      llm=llm, 
      prompt=prompt_template,
      verbose=False
    )
    start_time = time.time()
    urls = article_picker_chain.predict(response_str=response_str, query=query)
    url_list = json.loads(urls)
    print("--- Best Articles: %s seconds ---" % (time.time() - start_time))
    return url_list
  

def remove_tags(html):
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script']):
        data.decompose()
    return ' '.join(soup.stripped_strings)


def get_content_from_urls(urls):
  contents = []
  headers = { 'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0' }
  start_time = time.time()
  for url in urls:
    response = requests.get(url, headers=headers)
    contents.append(BeautifulSoup(response.text, features="html.parser").get_text())
  print("--- Content: %s seconds ---" % (time.time() - start_time))
  return contents
  
  
def summarize(llm, data, query):
  start_time = time.time()
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, 
    chunk_overlap=200, 
    length_function=len
  )
  text_docs = text_splitter.create_documents(data)
  print("--- Splitter: %s seconds ---" % (time.time() - start_time))

  template = """
  TEXT:
  {text}
  
  INSTRUCTIONS:
  You are a world class journalist, and you will try to summarize the text above in order to create a response about the query "{query}".
  Please folow all of the following rules:
  1. Make sure the content is informative with good data.
  2. Make sure the content is not too long. It should be no more than 500 words.
  3. The content should address the query: "{query}".
  4. The content needs to be written in a way that is easy to read and understand.
  
  SUMMARY:
  """
  prompt_template = PromptTemplate(input_variables=["text", "query"], template=template)
  summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
  summaries = []
  start_time = time.time()
  docs_n = len(text_docs)
  for chunk_i, chunk in enumerate(text_docs):
    summary = summarizer_chain.predict(text=chunk, query=query)
    summaries.append(summary)
    print(f"--- Finished {chunk_i + 1} out of {docs_n}")
  
  print("--- Summarize: %s seconds ---" % (time.time() - start_time))
  return summaries
  
  
def generate_response(llm, summaries, query):
  summary = str(summaries)
  template = """
  CONTEXT:
  {summary}
  
  INSTRUCTIONS:
  You are a world class journalist. The above text is some context about the query "{query}".
  Please write a response to the query "{query}" using the context above and following all the rules below:
  1. The response needs to be informative with good data.
  2. The response needs to be around 500 words.
  3. The response needs to address the query "{query}".
  4. The response needs to be easy to read and understand.
  
  RESPONSE:
  """
  prompt_template = PromptTemplate(input_variables=["summary", "query"], template=template)
  response_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
  start_time = time.time()
  response = response_chain.predict(summary=summary, query=query)
  print("--- Final Response: %s seconds ---" % (time.time() - start_time))
  return response
  
  
llm_mistral = OpenRouter(model="mistralai/mistral-7b-instruct")
llm_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
llm_davinci = OpenAI(model="text-davinci-003", temperature=0.7)
llm_curie = OpenAI(model="text-curie-001", temperature=0.7)

# query = "how are ai agents powered by llm?"
# optimized_query = optimize_query(llm_chat, query)
# print(optimized_query)
# search_response = search(optimized_query)
# urls = find_best_article_urls(llm_chat, search_response, optimized_query, top_n=False)
# print(urls)
# data = get_content_from_urls(urls)
# summaries = summarize(llm_davinci, data, optimized_query)
# print(generate_response(llm_chat, summaries, optimized_query))

# print(llm_mistral("Who are you?"))
print(llm_davinci("Who are you?"))