import pandas as pd, numpy as np

from typing import Union

from umap import UMAP
from hdbscan import HDBSCAN

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


from src.topic_modeling.document_preparation import clean_text
from src.config import modeling_config
from src.commons import Logger

def build_topic_model(all_stopwords: list[str]) -> BERTopic:
    """
    Assemble the BERTopic pipeline from its modular components.

    Components:
        - UMAP           : dimensionality reduction before clustering
        - HDBSCAN        : density-based clustering (produces topic assignments)
        - CountVectorizer: builds the vocabulary used for c-TF-IDF
        - ClassTfidfTransformer: computes per-topic keyword weights
        - representation_model: refines keyword selection after c-TF-IDF

    Note: calculate_probabilities=False avoids the O(n²) soft-membership
    computation that causes the pipeline to appear frozen on large datasets.
    """
    umap_model = UMAP(
        n_neighbors=modeling_config['N_NEIGHBORS'],
        n_components=modeling_config['N_COMPONENTS'],
        min_dist=0.0,          # 0.0 packs points tightly — good for clustering
        metric="cosine",
        random_state=modeling_config['SEED'],
        low_memory=False,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=modeling_config['MIN_CLUSTER_SIZE'],
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",   # produces more compact topics
        prediction_data=True,             # required for transform() on new docs
    )

    vectorizer_model = CountVectorizer(
        stop_words=all_stopwords,
        min_df=2,              # word must appear in at least 2 documents
        max_df=0.90,           # ignore words in more than 90 % of documents
        ngram_range=(1, 2),    # unigrams + bigrams to capture phrases
        strip_accents=None,    # keep Spanish accent characters
    )

    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,   # penalizes words shared across many topics
        bm25_weighting=True,          # BM25 outperforms classical TF-IDF here
    )

    representation_model = [
        KeyBERTInspired(),                        # semantic keyword selection
        MaximalMarginalRelevance(diversity=0.4),  # diversifies the keyword set
    ]

    model = BERTopic(
        embedding_model=modeling_config['EMBEDDING_MODEL_NAME'],
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model, # pyright: ignore[reportArgumentType]
        top_n_words=modeling_config['TOP_N_WORDS'],
        calculate_probabilities=False,   # set True only if you need soft probs
        verbose=True,
        language="multilingual",
    )

    return model

def train(
    topic_model: BERTopic,
    docs: list[str],
    embeddings: np.ndarray,
    logger: Logger
) -> tuple[list[int], Union[np.ndarray, None]]:
    """
    Fit BERTopic on the pre-computed embeddings.
    Returns the topic assignment per document and the probability array.
    """
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info[topic_info["Topic"] != -1])
    n_noise    = sum(t == -1 for t in topics)

    logger.info(f"  Topics found (excl. noise): {n_topics}")
    logger.info(f"  Documents in noise topic : {n_noise} ({n_noise / len(topics) * 100:.1f} %)")

    return topics, probs

def evaluate(
    topic_model: BERTopic,
    docs: list[str],
    embeddings: np.ndarray,
    topics: list[int],
    logger: Logger
) -> dict:
    """
    Compute three complementary quality metrics:

    1. CV Coherence (Gensim)
        Measures how semantically related the top words of each topic are.
        Range 0–1; values above 0.4 are generally considered acceptable.

    2. Silhouette Score (sklearn)
        Measures cluster compactness and separation in the 2D UMAP projection.
        Range -1 to 1; values above 0.2 indicate reasonable separation.

    3. Per-topic metrics DataFrame
        Reports document count, corpus weight, and lexical diversity per topic.

    Returns a dict with all computed values.
    """

    # --- 6.1  CV Coherence ---
    topic_words = {
        tid: [w for w, _ in topic_model.get_topic(tid)] # type: ignore
        for tid in topic_model.get_topics()
        if tid != -1
    }
    tokenized  = [doc.split() for doc in docs]
    gensim_dict = Dictionary(tokenized)

    coherence_cv = None
    try:
        cm = CoherenceModel(
            topics=list(topic_words.values()),
            texts=tokenized,
            dictionary=gensim_dict,
            coherence="c_v",
        )
        coherence_cv = cm.get_coherence()
        logger.info(f"  CV Coherence : {coherence_cv:.4f}")
    except Exception as e:
        logger.error(e)

    # --- 6.2  Silhouette Score ---
    non_noise_idx = [i for i, t in enumerate(topics) if t != -1]
    silhouette    = None

    if len({topics[i] for i in non_noise_idx}) >= 2:
        umap_2d    = UMAP(n_components=2, random_state=modeling_config['SEED'], metric="cosine")
        emb_2d     = umap_2d.fit_transform(embeddings[non_noise_idx])
        labels_2d  = [topics[i] for i in non_noise_idx]
        silhouette = silhouette_score(emb_2d, labels_2d, metric="euclidean") # pyright: ignore[reportArgumentType]
        logger.info(f"  Silhouette   : {silhouette:.4f}")
    else:
        logger.info("  Silhouette   : not enough topics")

    # --- 6.3  Per-topic metrics ---
    rows = []
    for tid, words in topic_words.items():
        n_docs = sum(1 for t in topics if t == tid)
        rows.append({
            "topic_id"  : tid,
            "n_docs"    : n_docs,
            "weight_%"  : round(n_docs / len(topics) * 100, 2),
            "diversity" : round(len(set(words)) / len(words), 4) if words else 0,
            "top_words" : ", ".join(words[:5]),
        })
    df_per_topic = pd.DataFrame(rows).sort_values("n_docs", ascending=False)
    logger.info(f"  Per-topic metrics:{df_per_topic.to_string(index=False)}")

    return {
        "coherence_cv" : coherence_cv,
        "silhouette"   : silhouette,
        "df_per_topic" : df_per_topic,
    }

def predict_new_documents(
    model_dir: str,
    embedding_model: SentenceTransformer,
    topic_labels: dict[int, str],
    new_texts: list[str]
) -> pd.DataFrame:
    """
    Load the saved BERTopic model and classify a list of new Spanish texts.
    Applies the same cleaning pipeline used during training before predicting.
    Returns a DataFrame with each text and its predicted topic.
    """
    loaded_model   = BERTopic.load(model_dir, embedding_model=modeling_config['EMBEDDING_MODEL_NAME'])
    texts_clean    = [clean_text(t) for t in new_texts]
    new_embeddings = embedding_model.encode(texts_clean, normalize_embeddings=True)

    new_topics, _ = loaded_model.transform(texts_clean, embeddings=new_embeddings.numpy())

    return pd.DataFrame({
        "document"    : new_texts,
        "topic_id"    : new_topics,
        "topic_label" : [topic_labels.get(t, "Noise") for t in new_topics],
    })