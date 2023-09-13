PVENUE_FIELDS_FOR_SEARCH = ['summary', 'description']

# Old index with the new BAAI model
PINECONE_ENV = "us-west1-gcp-free"
PINECONE_INDEX_NAME = "hybrid-venue-search"
# SEMANTIC SEARCH MODEL DETAILS
MODEL_NAME_NEW = "BAAI/bge-base-en"
MODEL_NEW_INSTRUCTION = "Represent this sentence for searching relevant passages:"

# new index with the old MPNET model
PINECONE_ENV_NEW = "gcp-starter"
PINECONE_INDEX_NAME_NEW = "hybrid-section-chunked"
# SEMANTIC SEARCH MODEL DETAILS
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

