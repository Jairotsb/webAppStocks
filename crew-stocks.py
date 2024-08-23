#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Installation

#get_ipython().system('pip install yfinance==0.2.41')
#get_ipython().system('pip install crewai==0.28.8')
#get_ipython().system("pip install 'crewai[tools]'")
#get_ipython().system('pip install langchain==0.1.20')
#get_ipython().system('pip install langchain-openai==0.1.7')
#get_ipython().system('pip install langchain-community==0.0.38')
#get_ipython().system('pip install duckduckgo-search==5.3.0')


# In[2]:


# IMPORT LIBS
import json
import os
from datetime import datetime




import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchResults

from IPython.display import Markdown


import streamlit as st

# In[3]:


# Creating YAHOO Finance Tool

def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2022-08-20", end="2024-08-20")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticket} from the last year about a specific stock from Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)


# In[4]:


# IMPORTING LLM GPT OPENAI
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY'] 

llm = ChatOpenAI()


# In[5]:


stockPriceAnalyst = Agent(
    role="Senior stock Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory=""" 
        You're a highly experienced in analyzing the price of an specific stock and make predictions about its future price.
    """, 
    verbose=True,
    llm=llm,
    max_iter=5, 
    allow_delegation=False,
    tools=[yahoo_finance_tool]
)


# In[6]:


get_stock_price = Task(
    description= "Analyze the stock {ticket} price history and create a trend analyses for up, down sideways",
    expected_output="""" Specify the current trend stock price - up, down or sideways. eg. stock="AAPL, price UP" """,
    agent=stockPriceAnalyst
)


# In[7]:


# Importing a tool of search

search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


# In[8]:


newsAnalyst = Agent(
    role="Stock News Analyst",
     goal="""
     Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down, or sideways
     with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.
     """,
    backstory=""" 
        You're highly experienced in analyzing the market trends and news have trecked asset for more then 10 years. 
        
        You're also master level analysts in the tradicional markets and have understanding of human psychology.
        
        You understand news, theirs titles and information, but you look at those with a health dose of skepticism.
        
        You consider also the source of the news articles.
        
    """, 
    verbose=True,
    llm=llm,
    max_iter=10, 
    allow_delegation=False,
    tools=[search_tool]
)


# In[9]:


get_news = Task(
    description= f"""
    Take the stock and always include BTC to it (if not request). 
    Use the search tool to search each one individually.
    
    The current date is {datetime.now()}
    Compose the results into a helpful report.
    """,
    expected_output=""""  
        A summary of the overall market and one sentence summary for each request asset.
        Include a fear/greed score for each asset based on the news. Use format:
        <STOCK ASSET>
        <SUMMARY BASED ON NEWS>
        <TREND PREDICTION>
        <FEAR/GREED SCORE>
    """,
    agent=newsAnalyst
)



# In[10]:


stockAnalystWrite = Agent(
    role="Senior stock analyst writer",
    goal="Analyze the trends price and news and write an insightfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend",
    backstory=""" 
        You're widely accepted as the best stock analyst in the market. 
        You understand complex concepts and create compelling stories and narratives that resonate 
        with wider audiences.
    
        You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. You're able to hold multiple opinions when analyzing anything.
    """, 
    verbose=True,
    llm= llm,
    max_iter=5, 
    memory=True,
    allow_delegation=True,
    tools=[yahoo_finance_tool]
)


# In[11]:


writeAnalyses = Task(
    description = """ 
    Use The stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
    that is brief and highlights the most important points. 
    
    Focus on the stock price trend, news and fear/greed  score. What are near future consideration?
    
    Include the previous analyses of stock trend and news summary.

    """,
    expected_output="""
        An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:
        
        - 3 bullets executive summary 
        - Introduction - set the overall picture and spike up the interest 
        - main part provides the meat of the analysis including the news summary and fear/gree scores.
        - summary - key facts and concrete future trend prediction - up, down, or sideways.
    """,
    agent= stockAnalystWrite, 
    context= [get_stock_price, get_news]
)


# In[12]:


crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [get_stock_price, get_news, writeAnalyses],
    verbose = 2,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)




# In[ ]:


#results = crew.kickoff(inputs={'ticket': 'AAPL'})


#results['final_output']


# In[16]:

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of research:")
        st.write(results['final_output'])