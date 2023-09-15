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
nltk.download('wordnet')
nltk.download('stopwords')


# Load the sentence embedding model
# https://www.sbert.net/docs/pretrained_models.html (all-MiniLM-L6-v2 is a good trade-off; paraphrase-MiniLM-L3-v2 is fastest)
# mini_lm = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# mini_lm = SentenceTransformer('paraphrase-MiniLM-L3-v2')

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
    # query pinecone with the query parameters
    if not baai_model:
        environment=config.PINECONE_ENV_NEW
        pinecone_index_name = config.PINECONE_INDEX_NAME_NEW
        api_key = st.secrets["PINECONE_API_KEY_NEW"]
    elif not bm25:
        api_key = st.secrets["PINECONE_API_KEY"]
        environment=config.PINECONE_ENV
        pinecone_index_name = config.PINECONE_INDEX_NAME
    else:
        api_key = st.secrets["PINECONE_API_KEY_BM25"]
        environment=config.PINECONE_ENV_BM25
        pinecone_index_name = config.PINECONE_INDEX_NAME_BM25
    
    pinecone.init(api_key=api_key, environment=environment)
    index = pinecone.Index(pinecone_index_name)
    # print(sparse_vec)

    conditions = {
        'site_birthday_prominence' : {'$in': ['high', 'medium']},
        'zipcode': {'$in': zipc_list}
    }

    # print(dense_vec)
    # print(sparse_vec)

    res = []
    res = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        filter=conditions,
        include_metadata=True
    )
    # return search results as json
    return res


def rerank_search_results(results, user_location):
    """
    Re-Rank the search results based on various criteria.
    """
    ranked_results = []

    for result in results:
        print(result)
        result = results[result]
        # print(result)
        # Calculate the distance from the user-specified location
        # if result['lat'] == "":
        #     coords1 = get_coordinates_from_zipcode(result['zipcode'])
        # else:
        #     coords1 = (result['lat'],result['long'])
        # coords2 = get_coordinates_from_zipcode(user_location)
        # distance = calculate_car_distance(coords1, coords2)
        # result['distance'] = distance
        # distance = distance/1000
        bp = result['birthday']
        if bp == 'none':
            bp = 0
        elif bp == 'low':
            bp = 1
        elif bp == 'medium':
            bp = 2
        else:
            bp = 3
        if result['ratings'] != "":
            # Calculate the score based on different criteria
            score = (
                    0.8 * result['score'] +  # Distance (weighted)
                    0 * bp +  # Site Birthday Prominence (weighted)
                    0 * result['ratings']  # rating (weighted)
            )
        else:
            score = (
                    0.8 * result['score'] +  # Distance (weighted)
                    0 * bp  # Site Birthday Prominence (weighted)
            )

        ranked_results.append({'result': result, 'score': score})

    # Sort the ranked results based on the score in descending order
    ranked_results.sort(key=lambda x: x['score'], reverse=True)

    return ranked_results

def closest_token_match(text_segment, search_query):
    # uses tokens from the search query and returns the sentence that contains most tokens
    sentences = re.split(r'[.!?]', text_segment)
    sentences = [s.strip() for s in sentences if s.strip()]

    max_matched_tokens = 0
    best_matching_sentence = ""

    for sentence in sentences:
        tokens_in_sentence = sentence.lower().split()
        tokens_in_query = search_query.lower().split()
        matched_tokens = [token for token in tokens_in_sentence if token in tokens_in_query]
        
        if len(matched_tokens) > max_matched_tokens:
            max_matched_tokens = len(matched_tokens)
            best_matching_sentence = sentence

    return re.sub(r'\b(' + '|'.join(search_query.split()) + r')\b', r'**\1**', best_matching_sentence.lower())

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

# def closest_match_fast(text_segment, search_query_embedding):
    # uses a mini-lm to compare query with every sentence in the text_segment and find the one which is closest to the query
    # List of sentences in the text segment
    sentences = text_segment.split('. ')

    # Calculate cosine similarity between search query and each sentence
    similarities = []
    for sentence in sentences:
        sentence_embedding = mini_lm.encode([sentence])
        similarity = cosine_similarity(search_query_embedding, sentence_embedding)[0][0]
        similarities.append(similarity)

    # Find the index of the most similar sentence
    most_similar_index = similarities.index(max(similarities))

    # Get the most similar sentence
    most_similar_sentence = sentences[most_similar_index]

    return most_similar_sentence


# def closest_match(text_segment, search_query):
    sentences = text_segment.split('. ')

    # Encode the search query
    # search_query_embedding = model.encode([search_query])
    inputs = tokenizer(search_query, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    search_query_embedding = outputs.pooler_output.detach().numpy().flatten()

    # Calculate cosine similarity between search query and each sentence
    sentence_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        sentence_embedding = outputs.pooler_output.detach().numpy().flatten()
        sentence_embeddings.append(sentence_embedding)

    # Calculate dot product between search query and sentence embeddings
    similarity_scores = np.dot(sentence_embeddings, search_query_embedding)

    # Find the index of the most similar sentence
    most_similar_index = np.argmax(similarity_scores)

    # Get the most similar sentence
    most_similar_sentence = sentences[most_similar_index]

    return most_similar_sentence


def search_venues(zip, radius, query, synonyms, baai_model, bm25, alpha):
    zipcode_list = find_nearby_zipcodes_pyzipcode(zip, radius)
    # print(f"searching for venues in {zipcode_list}")
    results = hybrid_query(query, zipcode_list, baai_model, bm25, alpha, top_k=100)
    # print(f"Number of results returned: {len(results['matches'])}")
    # print(results)

    # Encode the search query
    # search_query_embedding = mini_lm.encode([query])

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

        #if result['metadata']['site'] == "https://lightingartstudios.com/" or "bellevuestudio.com" in result['metadata']['site']:
        #    print(json.dumps(result, default=str, indent=4))

        # if "monsterminigolf.com" in result['metadata']['site']:
        #     print(json.dumps(result, default=str, indent=4))

    
    return refined_results

if __name__ == "__main__":
    bm25_model_filename = "bm25.pkl"
    
    st.title("Partify Venue Search")
    
    # Input fields
    interest = st.text_input("Child's Interest:", "")
    query = st.text_input("Search Query:", "martial arts")
    main_grid = st.container()
    with main_grid:
        cols = st.columns([1,1,1,4, 2])
        zip = cols[0].text_input("Zipcode:", "98045")
        radius = int(cols[1].text_input("Miles:", "20"))
        alpha = float(cols[2].text_input("Alpha:", value="0.75"))
        cols[4].write('')
        cols[4].write('')
        synonyms = cols[3].checkbox("Synomyms for rationale")
        baai_model = cols[3].checkbox("Use the BAAI model")
        bm25 = cols[3].checkbox("BM25 sparse embed with BAAI?")
    
        if not query and not interest:
            st.warning("Enter either the search query or child's interest. When both are given, search query is used")
        if not query and interest:
            query = interest    
    
        if query:
        # Search button
            if cols[4].button("Search"):
                # Perform search based on user inputs
                results = search_venues(zip, radius, clean_query(query), synonyms, baai_model, bm25, alpha)
                
                # Display results in a table
                # df = pd.DataFrame(results)
                # transposed_df = df.transpose()[['name', 'score', 'ratings', 'rationale', 'site', 'description']]  # Transpose the DataFrame to display as rows
                # st.table(transposed_df)

                search_results = pd.DataFrame(results).transpose().reset_index(drop=True)

                # Define CSS styles to limit row height
                row_height = "50px"  # Set your desired row height
                row_styles = [{"selector": "tr:hover td", "props": [("max-height", row_height)]}]

                # Display the search results table with custom styling
                st.dataframe(search_results[['name', 'max_score', 'avg_score', 'ratings', 'rationale', 'site', 'description']])

