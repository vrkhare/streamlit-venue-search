import re
import config
import string
import pickle
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import streamlit as st
from nltk.corpus import stopwords

@st.cache_data
def load_pretrained_model():
    print(f"Loading the pretrained model {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModel.from_pretrained(config.MODEL_NAME)
    print("done loading...")
    return tokenizer, model

tokenizer, model = load_pretrained_model()

@st.cache_data
def load_pretrained_model_baai():
    print(f"Loading the pretrained model {config.MODEL_NAME_NEW}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME_NEW)
    model = AutoModel.from_pretrained(config.MODEL_NAME_NEW)
    model.eval()
    print("done loading...")
    return tokenizer, model

tokenizer_baai, model_baai = load_pretrained_model_baai()

def dense_to_sparse(row):
    # Find the indices of non-zero entries
    indices = np.nonzero(row)[0]
    # Get the values of non-zero entries
    values = row[indices]
    return {'indices': indices.tolist(), 'values': values.tolist()}

def get_document_sparse_embeddings(document, model_filename):
    # this will get the sparse embeddings for one document at the time
    # can be used for queries at the time of retrieval   
    with open(model_filename, 'rb') as f:
        bm25 = pickle.load(f)
    return dense_to_sparse(bm25.transform([clean_query(document)])[0]) 

def build_dict(input_batch):
    # store a batch of sparse embeddings
    sparse_emb = []
    # print(input_batch)
    # iterate through input batch
    indices = []
    values = []
    # convert the input_ids list to a dictionary of key to frequency values
    d = dict(Counter(input_batch))
    for idx in d:
        indices.append(idx)
        values.append(float(d[idx]))
    # print(indices, values)
    sparse_emb = {'indices': indices, 'values': values}

    # return sparse_emb list
    return sparse_emb

def clean_query(qry):
    # List of common words to remove
    stop_words = set(stopwords.words('english'))

    # List of punctuations to remove
    punctuations = set(string.punctuation) 

    # Tokenize query
    query_tokens = qry.split() 

    # Split off trailing punctuation
    query_tokens = [re.sub(r'([^\s\w]$)','',token) for token in query_tokens]

    # Filter tokens
    filtered_tokens = [token for token in query_tokens if token not in stop_words and token not in punctuations]

    # Join filtered tokens 
    cleaned_query = " ".join(filtered_tokens)

    return cleaned_query

def generate_sparse_vectors(context_batch, baai_model):
    # from https://www.pinecone.io/learn/hybrid-search-intro/
    # Note that the generate_sparse_vectors method for creating sparse
    # vectors is not optimal. We recommend using either BM25 or SPLADE sparse vectors.
    # [9/15] this is now implemented in get_{corpus|document}_sparse_embeddings()

    # create batch of input_ids
    if not baai_model:
        inputs = tokenizer(
            context_batch, padding=True,
            truncation=True,
            max_length=512
        )['input_ids']
    else:
        inputs = tokenizer_baai(
            context_batch, padding=True,
            truncation=True,
            max_length=512
        )['input_ids']        
    #    print(inputs)
    # create sparse dictionaries
    sparse_embeds = build_dict(inputs)
    return sparse_embeds
