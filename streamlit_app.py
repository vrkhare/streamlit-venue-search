import os
import sys
import re
import json
import nltk

from nltk.corpus import wordnet 
from nltk.stem import PorterStemmer
import torch

import streamlit as st
import pandas as pd
import numpy as np

import pinecone
from venuehybridsearch import generate_sparse_vectors, get_document_sparse_embeddings, clean_query, tokenizer, model, tokenizer_baai, model_baai
import config
from pyzipcode import ZipCodeDatabase

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# Stemmer
stemmer = PorterStemmer()

@st.cache_data
def download_nltk_package():
    nltk.download('wordnet')
    nltk.download('stopwords')

download_nltk_package()

def find_nearby_zipcodes_pyzipcode(zipcode, max_distance_miles):
    # ref https://stackoverflow.com/questions/35047031/could-i-use-python-to-retrieve-a-number-of-zip-code-within-a-radius
    zcdb = ZipCodeDatabase()
    in_radius = [z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, max_distance_miles)]
    # radius_utf = [x.encode('UTF-8') for x in in_radius]  # unicode list to utf list
    return in_radius

def hybrid_scale(dense, sparse, alpha: float):
    # check alpha value is in range
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    #  print(hdense,hsparse)
    return hdense, hsparse

def hybrid_query(question, zipc_list, baai_model, bm25, alpha, top_k):
    # https://www.pinecone.io/learn/vector-search-filtering/
    # convert the question into a sparse vector
    if not bm25:
        sparse_vec = generate_sparse_vectors(question, baai_model)
    else:
        sparse_vec = get_document_sparse_embeddings(question, bm25_model_filename)
    # convert the question into a dense vector
    if not baai_model:
        inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        vectr = outputs.pooler_output.detach().numpy().flatten()
    else:
        # inputs = tokenizer_baai(question, padding=True, truncation=True, return_tensors="pt")
        # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
        # ref for instruction: https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list
        inputs = tokenizer_baai(config.MODEL_NEW_INSTRUCTION+question, padding=True, truncation=True, return_tensors='pt')
        outputs = model_baai(**inputs)
        # Perform pooling. In this case, cls pooling.
        vectr = outputs[0][:, 0]
        # normalize embeddings
        vectr = torch.nn.functional.normalize(vectr, p=2, dim=1)[0]
    
    dense_vec = vectr.tolist()
    # scale alpha with hybrid_scale
    # print(f"query: {question}; sparse-vector: {sparse_vec}; dense-vector: {dense_vec}")
    dense_vec, sparse_vec = hybrid_scale(
        dense_vec, sparse_vec, alpha
    )
    
    api_key = st.secrets["PINECONE_API_KEY"]
    environment=config.PINECONE_ENV
    pinecone_index_name = config.PINECONE_INDEX_NAME
    namespace = ""
    if not baai_model:
        namespace = "M1_C3_"
    else:
        namespace = "M2_C3_"
    
    if not bm25:
        namespace += "S1"
    else:
        namespace += "S2"

    pinecone.init(api_key=api_key, environment=environment)
    index = pinecone.Index(pinecone_index_name)
    # print(sparse_vec)

    conditions = {
        'zipcode': {'$in': zipc_list}
    }
    
    res = []
    res = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        filter=conditions,
        include_metadata=True,
        namespace=namespace
    )
    # return search results as json
    return res

def closest_query_phrase_match(text_segment, search_query, synonyms=False):
    # TODO: the following only works with two word phrase so far
    # For each document (text_segment), highlight a word or a phrase from the search query 
    # (search_query)  that matches the best with the document. The 
    # match is is calculated based on the number of overlapping words and phrases. Match score 
    # between a document and a word or a phrase in the search query is calculated based on the 
    # number of overlapping characters. Single word matches contribute the length of word to 
    # the match score, where as two word matches contribute to 2 times the length of the phrase. 
    # Likewise, three word matches contribute to 3 times the length of the phrase, and so on. 
    # Also, while calculating the matches the original search query is expended using word 
    # synonyms.
    synonyms = set()
    
    search_query = clean_query(search_query)
    text_segment = clean_query(text_segment)
    
    if synonyms:
        for word in search_query.split():
            for syn in wordnet.synsets(word):
                for w in syn.lemmas():
                    synonyms.add(w.name())
        expanded_query = search_query + " " + " ".join(synonyms) 
    else:
        expanded_query = search_query
    
    # Tokenize 
    query_tokens = expanded_query.split()
    doc_tokens = text_segment.split()

    # Stem tokens
    
    doc_stems = [stemmer.stem(token).lower() for token in doc_tokens] 
    stemmed_doc = ' '.join(doc_stems)

    query_stems = []
    stem_to_query_map = {}
    for token in query_tokens:
        query_stems.append(stemmer.stem(token).lower())
        stem_to_query_map[stemmer.stem(token).lower()] = token

    # Calculate match scores
    match_scores = {}
    for i in range(len(query_stems)-1):
        phrase = " ".join(query_stems[i:i+2])
        phrase_orig = " ".join(query_tokens[i:i+2])
        matches = [phrase in stemmed_doc]
        if matches.count(True) > 0:
            match_scores[phrase_orig] = 2 * matches.count(True)

    for word in query_stems:
        matches = [word in doc_stems]  
        if matches.count(True) > 0:
            match_scores[stem_to_query_map[word]] = matches.count(True)

    # Get best match
    if match_scores:
        best_match = max(match_scores, key=match_scores.get)
    else:
        best_match = ""

    return best_match

def search_venues(zip, radius, query, synonyms, baai_model, bm25, alpha):
    zipcode_list = find_nearby_zipcodes_pyzipcode(zip, radius)
    # print(f"searching for venues in {zipcode_list}")
    results = hybrid_query(query, zipcode_list, baai_model, bm25, alpha, top_k=config.NUM_CHUNK_IN_RESULTS)
    # print(f"Number of results returned: {len(results['matches'])}")
    # print(results)

    refined_results = {}
    for result in results['matches']:
        doc_id = result['id'].split("_")[0]
        if doc_id not in refined_results:
            refined_results[doc_id] = {}
            refined_results[doc_id]['avg_score'] = result['score'] / result['metadata']['chunks']
            refined_results[doc_id]['max_score'] = result['score']
            for key, values in result['metadata'].items():
                refined_results[doc_id][key] = values
            refined_results[doc_id]['rationale'] = closest_query_phrase_match(result['metadata']['description'], query, synonyms)
        else:
            refined_results[doc_id]['avg_score'] += result['score'] / result['metadata']['chunks']
            if result['score'] > refined_results[doc_id]['max_score']:
                refined_results[doc_id]['max_score'] = result['score']
    
    return refined_results

if __name__ == "__main__":
    bm25_model_filename = "bm25.pkl"
    
    st.title("Partify Venue Search")
    
    # Input fields
    interest = st.text_input("Child's Interest:", "")
    query = st.text_input("Search Query:", "martial arts")
    main_grid = st.container()
    with main_grid:
        cols = st.columns([1,1,1,2,2,4])
        zip = cols[0].text_input("Zipcode:", "98045")
        radius = int(cols[1].text_input("Miles:", "20"))
        alpha = float(cols[2].text_input("Alpha:", value="0.75"))
        synonyms = cols[5].checkbox("Synomyms for rationale")
        model_name = cols[3].radio("Language Model", ["bge", "mpnet"])
        if model_name == "bge":
            baai_model = True
        else: 
            baai_model = False
        sparse_embed = cols[4].radio("Sparse Embedding", ["bm25", "counter"])
        if sparse_embed == "bm25":
            bm25 = True
        else:
            bm25 = False
    
        if not query and not interest:
            st.warning("Enter either the search query or child's interest. When both are given, search query is used")
        if not query and interest:
            query = interest    
    
        if query:
        # Search button
            if cols[5].button("Search"):
                # Perform search based on user inputs
                results = search_venues(zip, radius, clean_query(query), synonyms, baai_model, bm25, alpha)
                search_results = pd.DataFrame(results).transpose().reset_index(drop=True)

                # Display the search results table with custom styling
                st.dataframe(search_results[['name', 'max_score', 'avg_score', 'ratings', 'rationale', 'site', 'description']])

