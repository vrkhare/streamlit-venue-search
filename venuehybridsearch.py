import config
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import streamlit as st


@st.cache_data
def load_pretrained_model():
    print(f"Loading the pretrained model {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModel.from_pretrained(config.MODEL_NAME)
    print("done loading...")
    return tokenizer, model

tokenizer, model = load_pretrained_model()


# https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token#sentencetransformers

def chunk_text_v2(text, ch_size):
    # note: CharacterTextSplitter chunks text into chunks of character size `ch_size` not based on tokens
    # split by tokens/ ref: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token
    # https://sbert.net/docs/pretrained_models.html
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=ch_size, chunk_overlap=0, word_level=True)

    for chunk in splitter.split_text(text):
        source_chunks.append(chunk)
    return source_chunks

def chunk_text(text, ch_size):
    # https://sbert.net/docs/pretrained_models.html
    # https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token#sentencetransformers
    source_chunks = []
    splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, model_name=config.MODEL_NAME)
    # print(f"maximum_tokens_per_chunk: {splitter.maximum_tokens_per_chunk}")
    # print(f"tokens in text to split: {splitter.count_tokens(text=text)}")
    # print(text)
    source_chunks = splitter.split_text(text=text)
    # for c in source_chunks:
    #    print(f"1. {c}")
    return source_chunks


def chunk_text_old(text, ch_size):
    # sentences = sent_tokenize(text)
    tokens = text.split(' ')
    chunks = []
    current_chunk = ""
    current_chunk_size = 0
    for token in tokens:
        current_chunk += token + " "
        current_chunk_size += 1

        if current_chunk_size >= ch_size:
            chunks.append(current_chunk)
            current_chunk = ""
            current_chunk_size = 0

    if current_chunk:
        chunks.append(current_chunk)
    # print(chunks)
    return chunks


def sparsecreate(record):
    """
    Creating texts which would be stored in sparse matrix
    TODO: updated to include meta.keywords on 8/7. hasn't been tested yet
    """
    keyword_string = ''
    # check if meta field has the sub-field keywords. This is important since it
    # is specified by the business website itself
    if 'meta' in record and record['meta'] is not None and 'keywords' in record['meta']:
        meta_keywords = record['meta']['keywords']
        if isinstance(meta_keywords, str):
            keyword_string += meta_keywords + ' '
        elif isinstance(meta_keywords, list):
            for x in meta_keywords:
                keyword_string += x + ' '
    # now add content from tags_dict
    tags_dict = record.get('tags', {})
    for key, value in tags_dict.items():
        # print(value)
        if isinstance(value, str):
            keyword_string += value + ' '
        else:
            for x in value:
                keyword_string += x + ' '
    return keyword_string


def create_doc_view(record):
    text = ''
    # first add name
    name_field = ''
    if 'name' in record and record['name'] is not None and len(record['name'])>0:
        name_field = record['name']
    else:
        url_dict = record.get('url', {})
        if 'title' in url_dict and url_dict['title'] is not None and len(url_dict['title'])>0:
            name_field = url_dict['title']

    if len(name_field) > 0:
        text += f"# Name\n{name_field}\n\n"
    # print(f"text after adding url_dict title: {text}")

    # now get all the keywords
    keywords = sparsecreate(record)
    if len(keywords)>0:
        text += f"# Keywords\n{keywords}\n\n"

    if 'summary' in record and record['summary'] is not None and len(record['summary'])>0:
        text += f"# Summary\n{str(record['summary'])}\n\n"
    
    if 'description' in record and record['description'] is not None and len(record['description'])>0:
        text += f"# Description\n{str(record['description'])}\n\n"
    else:
        meta_dict = record.get('meta', {})
        if 'description' in meta_dict and meta_dict['description'] is not None and len(meta_dict['description'])>0:
            text += f"# Description\n{meta_dict['description']}\n\n"
        

    # print(meta_dict)
    # print(f"text after adding meta_dict strings: {text}")

    # print(f"record['site_birthday_prominence']: {record['site_birthday_prominence']}")
    if record['site_birthday_prominence'] == "high":
        print("----------------------------")
        print(text)
        print("----------------------------")
    
    return text


# def vectorize_dictionary(record, chunk_size):
    """
    Returns vectors of the chunks of the data in each venue
    """
    chunks = []
    text = create_doc_view(record)
    chunks = chunk_text(text, chunk_size)

    vectors = []
    for chunk in chunks:
        inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        vectr = outputs.pooler_output.detach().numpy().flatten()
        vectors.append(vectr)

    address_dict = record.get('address', {})
    if address_dict is None:
        address_dict = {}

    if 'coordinates' in address_dict:
        lat = address_dict['coordinates']['lat']
        long = address_dict['coordinates']['lng']
    else:
        lat = long = ""

    if lat is None:
        lat = ""
    if long is None:
        long = ""

    if 'zip_code' in address_dict:
        zipcode = address_dict['zip_code']
        if zipcode is None:
            zipcode = ""
    else:
        zipcode = ""

    hours_dict = record.get('hours', {})
    days_closed = []
    for key, value in hours_dict.items():
        if isinstance(value, str):
            text = str(value)
            if text.lower() == 'closed' or text is None or text == '':
                days_closed.append(key)

    rating = record.get('ratings', {})
    if rating is None:
        rating = ""
    elif rating['y_rating'] is not None and rating['g_rating'] is not None:
        rating = (rating['y_rating'] * rating['y_num_reviews'] + rating['g_rating'] * rating['g_num_reviews']) / (
                    rating['y_num_reviews'] + rating['g_num_reviews'])
    elif rating['y_rating'] is not None:
        rating = rating['y_rating']
    elif rating['g_rating'] is not None:
        rating = rating['g_rating']    
    else:
        rating = ""

    if record['site_birthday_prominence'] is None:
        record['site_birthday_prominence'] = 'none'

    url_dict = record.get('url', {})
    if url_dict is None:
        url_dict = {}
    
    url = ""
    if 'full_url' in url_dict:
        url = url_dict['full_url']
        if url is None:
            url = ""


    # print(url_dict['full_url'])

    meta_data = {'site': url_dict['full_url'], 'name': str(record['name']), 'zipcode': zipcode,
                "DaysClosed": days_closed, 'lat': lat, 'long': long, 'ratings': rating,
                'birthday': record['site_birthday_prominence'], 'description': str(record['description']),
                'chunks': len(vectors)}
    # metadata ={}
    return vectors, meta_data


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


def generate_sparse_vectors(context_batch):
    # TODO: from https://www.pinecone.io/learn/hybrid-search-intro/
    # Note that the generate_sparse_vectors method for creating sparse
    # vectors is not optimal. We recommend using either BM25 or SPLADE sparse vectors.
    # However, to use either of these methods, we need corpus of documents.

    # create batch of input_ids
    inputs = tokenizer(
        context_batch, padding=True,
        truncation=True,
        max_length=512
    )['input_ids']
    #    print(inputs)
    # create sparse dictionaries
    sparse_embeds = build_dict(inputs)
    return sparse_embeds
