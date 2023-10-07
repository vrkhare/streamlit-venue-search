PINECONE_ENV = "us-west1-gcp-free"
PINECONE_INDEX_NAME = "hybrid-venue-search"
PINECONE_INDEX_NAME_BM25 = "hybrid-venue-bm25"
# SEMANTIC SEARCH MODEL DETAILS
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME_NEW = "BAAI/bge-base-en-v1.5"
MODEL_NEW_INSTRUCTION = "Represent this sentence for searching relevant passages:"
NUM_CHUNK_IN_RESULTS = 200