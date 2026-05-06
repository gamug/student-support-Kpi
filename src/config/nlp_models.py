import spacy, torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from src.config import modeling_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("es_core_news_md")

sentiment_model = pipeline(
        "sentiment-analysis",  # type: ignore
        model=modeling_config['SENTIMENT_MODEL_NAME'],
        device=device
    ) # type: ignore

embedding_model = SentenceTransformer(modeling_config['EMBEDDING_MODEL_NAME'], device=device) # type: ignore