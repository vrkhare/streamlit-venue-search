import nltk

from nltk.corpus import wordnet 
import time
from nltk.stem import PorterStemmer

import streamlit as st
import pandas as pd
import numpy as np

import pinecone
from venuehybridsearch import generate_sparse_vectors, get_document_sparse_embeddings, clean_query, tokenizer, model, tokenizer_baai, model_baai
import config
from pyzipcode import ZipCodeDatabase

from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# Stemmer
stemmer = PorterStemmer()

pre_db_call_time = db_call_time = post_db_call_time = total_time = time.time()
start_time = end_time = db_call_start = db_call_end = 0.0


@st.cache_resource
def init_pinecone_db(baai_model, bm25, fixed_length_chunks):
    environment=config.PINECONE_ENV
    api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_index_name = config.PINECONE_INDEX_NAME
    # api_key = st.secrets["PINECONE_API_KEY_BM25"]
    # pinecone_index_name = config.PINECONE_INDEX_NAME_BM25
    namespace = ""
    if not baai_model:
        namespace = "M1_"
    else:
        namespace = "M2_"
    
    if not fixed:
        namespace += "C3_"
    else:
        namespace += "C2_"
    if not bm25:
        namespace += "S1"
    else:
        namespace += "S2"

    pinecone.init(api_key=api_key, environment=environment)
    
    index = pinecone.Index(pinecone_index_name)

    return index, namespace

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
        # fallback for out of vocabulary words
        if len(sparse_vec['indices']) ==0:
            sparse_vec = get_document_sparse_embeddings("birthday", bm25_model_filename)
        
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
        # vectr = outputs[0][:, 0]
        # normalize embeddings
        # vectr = torch.nn.functional.normalize(vectr, p=2, dim=1)[0]

        vectr = outputs[0][:, 0].detach().numpy()  # Convert the PyTorch tensor to a NumPy array
        # Normalize embeddings using NumPy
        vectr = vectr / np.linalg.norm(vectr, ord=2, axis=1, keepdims=True)
        # If you want to access the first normalized vector (assuming there's only one)
        vectr = vectr[0]
    
    dense_vec = vectr.tolist()
    # scale alpha with hybrid_scale
    # print(f"query: {question}; sparse-vector: {sparse_vec}; dense-vector: {dense_vec}")
    dense_vec, sparse_vec = hybrid_scale(
        dense_vec, sparse_vec, alpha
    )
    # print(sparse_vec)

    conditions = {
        'zipcode': {'$in': zipc_list}
    }
    
    res = []
    global db_call_start, db_call_end
    db_call_start = time.time()
    res = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        filter=conditions,
        include_metadata=True,
        namespace=namespace
    )
    db_call_end = time.time()
    # return search results as json
    return res

def get_query_tokens(synonyms, search_query):
    syn_set = set()
    syn_to_query_map = {}
    orig_query_len = len(search_query.split())

    if synonyms:
        for word in search_query.split():
            for syn in wordnet.synsets(word):
                for w in syn.lemmas():
                    syn_set.add(w.name())
                    syn_to_query_map[w.name()] = word
        expanded_query = search_query + " " + " ".join(syn_set) 
    else:
        expanded_query = search_query
    
    # Tokenize 
    query_tokens = expanded_query.split()

    # Stem tokens
    stem_to_query_map = {}
    query_stemmed_set = set()
    for i in range(len(query_tokens)):
        # unigrams
        token = query_tokens[i]
        stem = stemmer.stem(token).lower()
        query_stemmed_set.add(stem)
        query_word = token
        if token in syn_to_query_map:
            query_word = syn_to_query_map[token]
        stem_to_query_map[stem] = query_word
        #bigrams - only with original tokens
        if i < orig_query_len - 1:
            next_token = query_tokens[i+1]
            stem_next = stemmer.stem(next_token).lower()
            stem_bigram = " ".join([stem, stem_next])
            orig_bigram = " ".join([token, next_token])
            query_stemmed_set.add(stem_bigram)
            stem_to_query_map[stem_bigram] = orig_bigram
    return query_stemmed_set, stem_to_query_map


def closest_query_phrase_match(text_segment, query_stemmed_set, stem_to_query_map):
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
    text_segment = clean_query(text_segment)
    
    # Tokenize 
    doc_tokens = text_segment.split()

    # Stem tokens
    doc_stems = [stemmer.stem(token).lower() for token in doc_tokens] 
    # print(f"query_stemmed_set:{query_stemmed_set}")
    # print(f"doc_stems: {doc_stems}")

    # Calculate unigram and bi-gram frequencies in the doc
    doc_unigram_frequency_map = {}
    doc_bigram_frequency_map = {}
    for i in range(len(doc_stems)):
        token = doc_stems[i]
        if token in query_stemmed_set:
            if token in stem_to_query_map:
                query_token = stem_to_query_map[token] 
            else:
                query_token = token
            
            if query_token in doc_unigram_frequency_map:
                doc_unigram_frequency_map[query_token] += 1
            else:
                doc_unigram_frequency_map[query_token] = 1
            if i < len(doc_stems)-1:
                next_token = doc_stems[i+1]
                bigram = " ".join([token, next_token])
                if next_token in stem_to_query_map:
                    query_next_token = stem_to_query_map[next_token] 
                else:
                    query_next_token = next_token
                if bigram in query_stemmed_set:
                    if bigram in doc_bigram_frequency_map:
                        doc_bigram_frequency_map[bigram] += 1
                    else:
                        doc_bigram_frequency_map[bigram] = 1
                    # since the token and next are counted in bi-gram, remove them from unigram freq
                    doc_unigram_frequency_map[query_token] -= 1
                    if query_next_token in doc_unigram_frequency_map:
                        doc_unigram_frequency_map[query_next_token] -= 1
                    else:
                        doc_unigram_frequency_map[query_next_token] = -1
    
    # check if there is a bi-gram found
    if doc_bigram_frequency_map:
        best_match = stem_to_query_map[max(doc_bigram_frequency_map, key=doc_bigram_frequency_map.get)]
    elif doc_unigram_frequency_map:
        best_match_unigram = max(doc_unigram_frequency_map, key=doc_unigram_frequency_map.get)
        if best_match_unigram in stem_to_query_map:
            best_match = stem_to_query_map[best_match_unigram]
        else:
            best_match = best_match_unigram
    else:
        best_match = ""

    return best_match

def search_venues(zip, radius, query, synonyms, baai_model, bm25, alpha):
    zipcode_list = find_nearby_zipcodes_pyzipcode(zip, radius)
    # print(f"searching for venues in {zipcode_list}")
    results = hybrid_query(query, zipcode_list, baai_model, bm25, alpha, top_k=config.NUM_CHUNK_IN_RESULTS)
    # print(f"Number of results returned: {len(results['matches'])}")
    # print(results['matches'][0])

    # query_tokens, query_stems, stem_to_query_map = get_query_tokens(synonyms, clean_query(query))
    query_stemmed_set, stem_to_query_map = get_query_tokens(synonyms, clean_query(query))

    refined_results = {}
    for result in results['matches']:
        doc_id = result['id'].split("_")[0]
        if doc_id not in refined_results:
            refined_results[doc_id] = {}
            # refined_results[doc_id]['_id'] = result['metadata']['_id']
            # refined_results[doc_id]['avg_score'] = result['score'] / result['metadata']['chunks']
            refined_results[doc_id]['max_score'] = result['score']
            for key, values in result['metadata'].items():
                refined_results[doc_id][key] = values
            refined_results[doc_id]['rationale'] = closest_query_phrase_match(result['metadata']['description'], query_stemmed_set, stem_to_query_map)
        else:
            # refined_results[doc_id]['avg_score'] += result['score'] / result['metadata']['chunks']
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
        cols = st.columns([1,1,1,2,2,2,3])
        zip = cols[0].text_input("Zipcode:", "98045")
        radius = int(cols[1].text_input("Miles:", "200"))
        alpha = float(cols[2].text_input("Alpha:", value="0.75"))
        synonyms = cols[6].checkbox("Synomyms for rationale", value=True)
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
        
        chunking = cols[5].radio("Chunking", ["fixed", "markdown"])
        if chunking == "fixed":
            fixed = True
        else:
            fixed = False
    
        if not query and not interest:
            st.warning("Enter either the search query or child's interest. When both are given, search query is used")
        if not query and interest:
            query = interest    

        index, namespace = init_pinecone_db(baai_model, bm25, fixed)
        if query:
        # Search button
            if cols[6].button("Search"):
                # Perform search based on user inputs
                start_time = time.time()
                results = search_venues(zip, radius, clean_query(query), synonyms, baai_model, bm25, alpha)
                end_time = time.time()
                total_time = end_time - start_time
                pre_db_call_time = db_call_start - start_time
                db_call_time = db_call_end - db_call_start
                post_db_call_time = end_time - db_call_end

                print(f"{namespace}, {pre_db_call_time}, {db_call_time}, {post_db_call_time}, {total_time}")
                search_results = pd.DataFrame(results).transpose().reset_index(drop=True)

                # Display the search results table with custom styling
                st.dataframe(search_results[['name', '_id', 'address', 'max_score', 'ratings', 'partify_place_type', 'thumb_url', 'rationale', 'description']])

