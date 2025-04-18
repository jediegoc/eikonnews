#%%
import streamlit as st
import pandas as pd
import numpy as np
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as Pinecone_lang
from pinecone import Pinecone, ServerlessSpec
#%%
st.set_page_config(
    page_title="Home"    
)

st.write("# Q&A with news sources")

st.sidebar.success("Select an Option above")

st.markdown(
    """
    **This App provides the following functionalities:**
    - Q&A with news articles for selected companies
    - Q&A with news articles for a range of companies
    - Q&A with news articles for all companies
"""
)

#%% Load RAG
@st.cache_resource
def load_RAG():
    #Load env. variables
    os.environ['OPENAI_API_KEY']='sk-proj-n6q2SRr0XB7aeDN7gQwCT3BlbkFJQzztGFKlZv4RBYmDEhpX'
    os.environ['PINECONE_API_KEY']='pcsk_2Zwydu_5FZX16wb3hD2fnVLh9AHwUrhGPizs9GLR2jf2Gf8YRxvjYpgvXM6u7LEvjRy7Y6'
    #%%  

    pc=Pinecone(
        api_key=os.environ['PINECONE_API_KEY'],
        environment='gpc-starter'
    )
    embeddings = OpenAIEmbeddings()
    index_name = "refinitivnews-texts"
    index = pc.Index(index_name)
    docsearch = Pinecone_lang.from_existing_index(index_name, embeddings) 

    return docsearch

docsearch=load_RAG()
st.session_state.docsearch=docsearch