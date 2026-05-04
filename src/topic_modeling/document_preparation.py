
import nltk, os, re
import pandas as pd, numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from src.config import modeling_config
from src.commons import Logger


def load_data() -> list[str]:
    """
    Load raw Spanish documents from your data source.
    Returns a list of raw strings. Replace the demo block
    with Option A (CSV) or Option B (TXT) for real data.
    """

    # --- Option A: CSV ---
    # df = pd.read_csv("your_texts.csv")
    # return df["text_column"].dropna().tolist()

    # --- Option B: plain text file, one document per line ---
    # with open("texts.txt", "r", encoding="utf-8") as f:
    #     return [line.strip() for line in f if line.strip()]

    # --- Option C (DEMO): diverse sentences so embeddings are non-degenerate ---
    df = pd.read_excel(os.path.join('..', 'output', 'stundent_support_corpus.xlsx'), index_col=0)
    df = df[(df.sentiment_label=='NEG')&(df.sentence_subject=='PROGRAM')]
    docs = df.sentence.tolist()

    return docs

def build_stopwords() -> list[str]:
    """
    Combine NLTK Spanish stopwords with custom domain stopwords.
    Returns the merged list ready for CountVectorizer.
    """
    nltk.download("stopwords", quiet=True)
    base      = list(stopwords.words("spanish"))
    custom    = [
        "además", "aunque", "sino", "así", "entonces",
        "puede", "ser", "hacer", "tener", "estar", "haber",
        "dice", "dijo", "según", "año", "años", "día", "días",
    ]
    return list(set(base + custom))


def clean_text(text: str) -> str:
    """
    Clean a single Spanish document:
    - lowercase
    - remove URLs, @mentions, #hashtags
    - remove non-alphabetic characters (keeps á é í ó ú ü ñ)
    - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)        # URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # mentions / hashtags
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)        # non-Spanish chars
    text = re.sub(r"\s+", " ", text).strip()             # extra spaces
    return text


def preprocess(docs: list[str], logger: Logger) -> list[str]:
    """
    Apply clean_text to every document and drop texts that are
    too short to carry meaningful topic signal after cleaning.
    """
    cleaned  = [clean_text(d) for d in docs]
    filtered = [d for d in cleaned if len(d.split()) >= modeling_config['MIN_WORDS']]
    logger.info(f"  Documents after cleaning: {len(filtered)}")
    return filtered

def compute_embeddings(docs: list[str], logger: Logger) -> tuple[SentenceTransformer, np.ndarray]:
    """
    Load a multilingual SentenceTransformer and encode all documents.
    Returns both the model (needed later for inference) and the embeddings array.
    Pre-computing embeddings here avoids redundant encoding during fit_transform.
    """
    logger.info(f"  Loading embedding model: {modeling_config['EMBEDDING_MODEL_NAME']}")
    model      = SentenceTransformer(modeling_config['EMBEDDING_MODEL_NAME'])
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,   # L2 norm improves cluster quality
    )
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    return model, embeddings