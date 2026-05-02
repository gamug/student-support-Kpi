import spacy
from transformers import pipeline

nlp = spacy.load("es_core_news_md")

sentiment_model = pipeline(
        "sentiment-analysis", 
        model="pysentimiento/robertuito-sentiment-analysis",
        device=0
    )