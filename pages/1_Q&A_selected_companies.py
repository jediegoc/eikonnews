#%% Load libraries
import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import re
from IPython.display import display, Markdown
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.callbacks import BaseCallbackHandler
from Home import load_RAG

#%% Load docssearch if not initialized
if "docsearch" not in st.session_state:
    docsearch=load_RAG()
    st.session_state.docsearch=docsearch

#%% Read/adapt the prompts
DOCUMENT_PROMPT = """{page_content}
Date: {Date}
Headline: {Headline}
Company_Name: {Company_Name}
RIC: {RIC}
Source: {Source}
========="""

with open("files/QUESTION_PROMPT.txt", "r") as f:
    QUESTION_PROMPT = f.read()

# Create prompt template objects
document_prompt = PromptTemplate.from_template(DOCUMENT_PROMPT)
question_prompt = PromptTemplate.from_template(QUESTION_PROMPT)

#%% Configure the retriever with the input of user chosen companies
def retriever(chosen_companies):
    # Import RetrievalQAWithSourcesChain and ChatOpenAI
    class MyCustomHandler(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            st.session_state.output=str(response.generations[0][0])     

    llm = ChatOpenAI(model="gpt-4o", temperature=0,callbacks=[MyCustomHandler()],verbose=False)

    Year='2023'
    company_name='Codexis Inc'

    # Create the QA bot LLM chain
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        chain_type="stuff",
        llm=llm,
        chain_type_kwargs={
            "document_prompt": document_prompt,
            "prompt": question_prompt,
        },
        retriever=st.session_state.docsearch.as_retriever( 
        search_kwargs={#"score_threshold": 0.001,                                 
        "filter": {
                "Company_Name":{"$in":chosen_companies},                        
            },"k":40}),
        #search_kwargs={"k":400}
    )
    return qa_with_sources

#%% Read available companies
def read():
    companies=pd.read_csv("Companies_NYSE_NASDAQ_Pinecone.csv")
    return companies

companies=read()

#%% SEt visual elements for the page

try:
    st.set_page_config(
        page_title="Q&A selected companies"    
    )
except Exception:
    pass

st.markdown("# Q&A for selected companies")
st.sidebar.header("Q&A for selected companies" )

#Expander
expander = st.expander("Question Prompt")
prompt=expander.text_area(label="Question Prompt",value=QUESTION_PROMPT)
if prompt:
    question_prompt = PromptTemplate.from_template(prompt)
    
#Input Companies
chosen_companies=st.sidebar.multiselect("Choose a Company",companies['Company_Name'])
#Dataframe showing selected companies
st.dataframe(chosen_companies)
#Clear history button
button=st.sidebar.button("Clear chat history")
if button:
    st.session_state.messages = []
#Add a sidebar dataframe showing all companies    
st.sidebar.dataframe(companies['Company_Name'])
#%% Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.html(message["content"])
#Format messages
def format_markdown(answer1):
  #answer1=re.search('text=[\'"](.*?)\s*generation_info',answer1).group(1)
  #answer1=re.sub(r'\\n',"<br>",answer1)
  match = re.search(r"message=AIMessage\(content='(.*?)'\s*,\s*additional_kwargs=", answer1, re.DOTALL)
  if match:
      content_text = match.group(1)
      # Convert escaped newlines to actual HTML breaks
      content_text = content_text.replace("\\n", "<br>")
      return content_text
  else:
      return "AIMessage content not found."
  #return answer1

# Accept user input and run RAG
if prompt := st.chat_input("Ask a question. E.g. Which AI technologies have companies adopted?"):
    
    question = {'question':prompt+"Companies: "+', '.join(chosen_companies)}
    answer=retriever(chosen_companies).invoke(question)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response=""
        assistant_response = format_markdown(st.session_state.output)
        st.session_state.output=""
        # Simulate stream of response with milliseconds delay
        #for chunk in assistant_response.split():
        #    full_response += chunk + " "
        #    time.sleep(0.05)
        #    # Add a blinking cursor to simulate typing
        full_response += assistant_response
        message_placeholder.html(full_response + "▌")            
        #message_placeholder.html(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
