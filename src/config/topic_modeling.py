modeling_config = {
    'SEED': 42,
    'SENTIMENT_MODEL_NAME': "pysentimiento/robertuito-sentiment-analysis",
    'EMBEDDING_MODEL_NAME': "paraphrase-multilingual-MiniLM-L12-v2",
    'MIN_WORDS': 4,       # minimum words per document after cleaning
    'MIN_CLUSTER_SIZE': 15,      # HDBSCAN: minimum docs to form a topic
    'N_NEIGHBORS': 15,      # UMAP: local vs global structure balance
    'N_COMPONENTS': 10,      # UMAP: reduced dimensions before clustering
    'TOP_N_WORDS': 12      # keywords shown per topic
}