# =============================================================================
# TOPIC MODELING WITH BERTOPIC - SPANISH TEXTS
# Fully refactored into functions, executed inside if __name__ == "__main__"
# =============================================================================


# =============================================================================
# 0. CENTRALIZED IMPORTS
# =============================================================================

import json, nltk, os, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

from umap import UMAP
from hdbscan import HDBSCAN

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

import plotly.io as pio

warnings.filterwarnings("ignore")


# =============================================================================
# CONSTANTS
# =============================================================================

SEED                 = 42
OUTPUT_DIR           = Path("../bertopic_results")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_WORDS            = 4       # minimum words per document after cleaning
MIN_CLUSTER_SIZE     = 15      # HDBSCAN: minimum docs to form a topic
N_NEIGHBORS          = 15      # UMAP: local vs global structure balance
N_COMPONENTS         = 10      # UMAP: reduced dimensions before clustering
TOP_N_WORDS          = 12      # keywords shown per topic


# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================


# -----------------------------------------------------------------------------
# STEP 1 — Load data
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# STEP 2 — Preprocess
# -----------------------------------------------------------------------------

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


def preprocess(docs: list[str]) -> list[str]:
    """
    Apply clean_text to every document and drop texts that are
    too short to carry meaningful topic signal after cleaning.
    """
    cleaned  = [clean_text(d) for d in docs]
    filtered = [d for d in cleaned if len(d.split()) >= MIN_WORDS]
    print(f"  Documents after cleaning: {len(filtered)}")
    return filtered


# -----------------------------------------------------------------------------
# STEP 3 — Embeddings
# -----------------------------------------------------------------------------

def compute_embeddings(docs: list[str]) -> tuple[SentenceTransformer, np.ndarray]:
    """
    Load a multilingual SentenceTransformer and encode all documents.
    Returns both the model (needed later for inference) and the embeddings array.
    Pre-computing embeddings here avoids redundant encoding during fit_transform.
    """
    print(f"  Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model      = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,   # L2 norm improves cluster quality
    )
    print(f"  Embeddings shape: {embeddings.shape}")
    return model, embeddings


# -----------------------------------------------------------------------------
# STEP 4 — Build BERTopic model
# -----------------------------------------------------------------------------

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
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        min_dist=0.0,          # 0.0 packs points tightly — good for clustering
        metric="cosine",
        random_state=SEED,
        low_memory=False,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
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
        embedding_model=EMBEDDING_MODEL_NAME,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=TOP_N_WORDS,
        calculate_probabilities=False,   # set True only if you need soft probs
        verbose=True,
        language="multilingual",
    )

    return model


# -----------------------------------------------------------------------------
# STEP 5 — Train
# -----------------------------------------------------------------------------

def train(
    topic_model: BERTopic,
    docs: list[str],
    embeddings: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """
    Fit BERTopic on the pre-computed embeddings.
    Returns the topic assignment per document and the probability array.
    """
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info[topic_info["Topic"] != -1])
    n_noise    = sum(1 for t in topics if t == -1)

    print(f"  Topics found (excl. noise): {n_topics}")
    print(f"  Documents in noise topic : {n_noise} ({n_noise / len(topics) * 100:.1f} %)")
    print(f"\n  Top 5 topics by size:\n{topic_info.head(6).to_string(index=False)}")

    return topics, probs


# -----------------------------------------------------------------------------
# STEP 6 — Evaluate
# -----------------------------------------------------------------------------

def evaluate(
    topic_model: BERTopic,
    docs: list[str],
    embeddings: np.ndarray,
    topics: list[int],
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
        tid: [w for w, _ in topic_model.get_topic(tid)]
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
        print(f"  CV Coherence : {coherence_cv:.4f}")
    except Exception as e:
        print(f"  CV Coherence could not be computed: {e}")

    # --- 6.2  Silhouette Score ---
    non_noise_idx = [i for i, t in enumerate(topics) if t != -1]
    silhouette    = None

    if len(set(topics[i] for i in non_noise_idx)) >= 2:
        umap_2d    = UMAP(n_components=2, random_state=SEED, metric="cosine")
        emb_2d     = umap_2d.fit_transform(embeddings[non_noise_idx])
        labels_2d  = [topics[i] for i in non_noise_idx]
        silhouette = silhouette_score(emb_2d, labels_2d, metric="euclidean")
        print(f"  Silhouette   : {silhouette:.4f}")
    else:
        print("  Silhouette   : not enough topics")

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
    print(f"\n  Per-topic metrics:\n{df_per_topic.to_string(index=False)}")

    return {
        "coherence_cv" : coherence_cv,
        "silhouette"   : silhouette,
        "df_per_topic" : df_per_topic,
    }


# -----------------------------------------------------------------------------
# STEP 7 — Post-process
# -----------------------------------------------------------------------------

def postprocess(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    all_stopwords: list[str],
    n_topics_found: int,
) -> tuple[list[int], dict[int, str]]:
    """
    Three post-processing steps applied after training:

    1. reduce_topics  — merges similar topics until a target count is reached.
    2. set_topic_labels — assigns human-readable labels built from top keywords.
    3. update_topics  — refreshes keyword representation without re-training
                        UMAP/HDBSCAN (safe to call multiple times).

    Returns updated topic assignments and the label mapping dict.
    """

    # --- 7.1  Reduce number of topics ---
    target = max(5, n_topics_found // 2)
    print(f"  Reducing to ~{target} topics...")
    topic_model.reduce_topics(docs, nr_topics=target)
    topics_reduced = topic_model.topics_
    n_final = len(topic_model.get_topic_info().query("Topic != -1"))
    print(f"  Topics after reduction: {n_final}")

    # --- 7.2  Build and assign readable labels ---
    topic_labels = {}
    for tid in topic_model.get_topics():
        if tid == -1:
            continue
        top3 = [w for w, _ in topic_model.get_topic(tid)[:3]]
        topic_labels[tid] = f"T{tid}: {' | '.join(top3)}"

    topic_model.set_topic_labels(topic_labels)
    print(f"  Sample labels: {list(topic_labels.values())[:4]}")

    # --- 7.3  Update topic representation with trigrams ---
    topic_model.update_topics(
        docs,
        vectorizer_model=CountVectorizer(
            stop_words=all_stopwords,
            ngram_range=(1, 3),   # extend to trigrams after reduction
            min_df=2,
            max_df=0.85,
        ),
        representation_model=MaximalMarginalRelevance(diversity=0.5),
        top_n_words=10,
    )
    print("  Topic representation updated.")

    return topics_reduced, topic_labels


# -----------------------------------------------------------------------------
# STEP 8 — Build results DataFrame
# -----------------------------------------------------------------------------

def build_results_dataframe(
    docs: list[str],
    topics: list[int],
    topic_labels: dict[int, str],
) -> pd.DataFrame:
    """
    Combine documents, topic assignments, and labels into a single DataFrame
    that is ready for analysis and export.
    """
    return pd.DataFrame({
        "document"    : docs,
        "topic_id"    : topics,
        "topic_label" : [topic_labels.get(t, "Noise") for t in topics],
    })


# -----------------------------------------------------------------------------
# STEP 9 — Visualize
# -----------------------------------------------------------------------------

def visualize(
    topic_model: BERTopic,
    docs: list[str],
    embeddings: np.ndarray,
    n_topics_final: int,
) -> None:
    """
    Generate five interactive Plotly charts and save each as a standalone
    HTML file in OUTPUT_DIR. Open any of them in a browser to explore.

    Charts produced:
      - document_map.html       : 2D scatter of every document colored by topic
      - topic_barchart.html     : top keywords per topic side by side
      - similarity_heatmap.html : pairwise cosine similarity between topics
      - topic_hierarchy.html    : dendrogram showing topic relationships
      - intertopic_distance.html: bubble chart where distance = dissimilarity
    """

    def _save(fig, name: str) -> None:
        path = OUTPUT_DIR / f"{name}.html"
        fig.write_html(str(path))
        print(f"  Saved: {path}")

    _save(
        topic_model.visualize_documents(
            docs, embeddings=embeddings,
            hide_annotations=False,
            title="Document map by topic",
        ),
        "document_map",
    )

    _save(
        topic_model.visualize_barchart(
            top_n_topics=min(n_topics_final, 12),
            title="Top keywords per topic",
            n_words=8,
        ),
        "topic_barchart",
    )

    _save(
        topic_model.visualize_heatmap(
            title="Inter-topic similarity heatmap",
            width=800, height=800,
        ),
        "similarity_heatmap",
    )

    _save(
        topic_model.visualize_hierarchy(title="Topic hierarchy (dendrogram)"),
        "topic_hierarchy",
    )

    _save(
        topic_model.visualize_topics(title="Intertopic distance map"),
        "intertopic_distance",
    )


# -----------------------------------------------------------------------------
# STEP 10 — Export
# -----------------------------------------------------------------------------

def export(
    topic_model: BERTopic,
    df_results: pd.DataFrame,
    topic_labels: dict[int, str],
    eval_metrics: dict,
    topics_initial: list[int],
) -> None:
    """
    Write all results to OUTPUT_DIR:

      documents_with_topics.csv  : one row per document with topic + label
      topic_info.csv             : BERTopic's built-in topic metadata table
      topic_keywords.json        : keywords and scores for every topic
      evaluation_metrics.csv     : coherence, silhouette, and run metadata
      bertopic_model/            : serialized model (safetensors format)
    """

    # Documents CSV
    path = OUTPUT_DIR / "documents_with_topics.csv"
    df_results.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Documents CSV      : {path}")

    # Topic info CSV
    path = OUTPUT_DIR / "topic_info.csv"
    topic_model.get_topic_info().to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Topic info CSV     : {path}")

    # Keywords JSON
    keywords_dict = {}
    for tid in topic_model.get_topics():
        if tid == -1:
            continue
        keywords_dict[str(tid)] = {
            "label"   : topic_labels.get(tid, f"Topic {tid}"),
            "keywords": [
                {"word": w, "score": round(s, 4)}
                for w, s in topic_model.get_topic(tid)
            ],
        }
    path = OUTPUT_DIR / "topic_keywords.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keywords_dict, f, ensure_ascii=False, indent=2)
    print(f"  Keywords JSON      : {path}")

    # Evaluation metrics CSV
    n_topics_final = len(topic_model.get_topic_info().query("Topic != -1"))
    n_noise        = sum(1 for t in df_results["topic_id"] if t == -1)
    df_eval = pd.DataFrame([{
        "n_topics_initial" : len(set(topics_initial)) - (1 if -1 in topics_initial else 0),
        "n_topics_final"   : n_topics_final,
        "n_documents"      : len(df_results),
        "n_noise"          : n_noise,
        "pct_noise"        : round(n_noise / len(df_results) * 100, 2),
        "coherence_cv"     : round(eval_metrics["coherence_cv"], 4) if eval_metrics["coherence_cv"] else None,
        "silhouette_score" : round(eval_metrics["silhouette"], 4)   if eval_metrics["silhouette"]   else None,
        "embedding_model"  : EMBEDDING_MODEL_NAME,
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M"),
    }])
    path = OUTPUT_DIR / "evaluation_metrics.csv"
    df_eval.to_csv(path, index=False)
    print(f"  Evaluation CSV     : {path}")

    # Serialized model
    path = OUTPUT_DIR / "bertopic_model"
    topic_model.save(
        str(path),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,   # embedding model is reloaded by name on load
    )
    print(f"  Model saved        : {path}")


# -----------------------------------------------------------------------------
# STEP 11 — Inference on new documents
# -----------------------------------------------------------------------------

def predict_new_documents(
    model_dir: Path,
    embedding_model: SentenceTransformer,
    topic_labels: dict[int, str],
    new_texts: list[str],
) -> pd.DataFrame:
    """
    Load the saved BERTopic model and classify a list of new Spanish texts.
    Applies the same cleaning pipeline used during training before predicting.
    Returns a DataFrame with each text and its predicted topic.
    """
    loaded_model   = BERTopic.load(str(model_dir), embedding_model=EMBEDDING_MODEL_NAME)
    texts_clean    = [clean_text(t) for t in new_texts]
    new_embeddings = embedding_model.encode(texts_clean, normalize_embeddings=True)

    new_topics, _ = loaded_model.transform(texts_clean, embeddings=new_embeddings)

    df_pred = pd.DataFrame({
        "document"    : new_texts,
        "topic_id"    : new_topics,
        "topic_label" : [topic_labels.get(t, "Noise") for t in new_topics],
    })

    print(df_pred.to_string(index=False))
    return df_pred


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Prevents Windows worker processes from restarting the pipeline.
    # On Windows, multiprocessing uses 'spawn', so every subprocess
    # re-imports this file. Without this guard the script restarts endlessly.

    pio.renderers.default = "browser"
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("TOPIC MODELING IN SPANISH WITH BERTOPIC")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading data...")
    docs_raw = load_data()
    print(f"  Raw documents loaded: {len(docs_raw)}")

    # ------------------------------------------------------------------
    print("\n[STEP 2] Preprocessing...")
    all_stopwords = build_stopwords()
    docs          = preprocess(docs_raw)

    # ------------------------------------------------------------------
    print("\n[STEP 3] Computing embeddings...")
    embedding_model, embeddings = compute_embeddings(docs)

    # ------------------------------------------------------------------
    print("\n[STEP 4] Building BERTopic model...")
    topic_model = build_topic_model(all_stopwords)

    # ------------------------------------------------------------------
    print("\n[STEP 5] Training...")
    topics, probs = train(topic_model, docs, embeddings)
    n_topics_found = len(topic_model.get_topic_info().query("Topic != -1"))

    # ------------------------------------------------------------------
    print("\n[STEP 6] Evaluating...")
    eval_metrics = evaluate(topic_model, docs, embeddings, topics)

    # ------------------------------------------------------------------
    print("\n[STEP 7] Post-processing...")
    topics_final, topic_labels = postprocess(
        topic_model, docs, topics, all_stopwords, n_topics_found
    )
    n_topics_final = len(topic_model.get_topic_info().query("Topic != -1"))

    # ------------------------------------------------------------------
    print("\n[STEP 8] Building results DataFrame...")
    df_results = build_results_dataframe(docs, topics_final, topic_labels)
    print(df_results.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[STEP 9] Generating visualizations...")
    visualize(topic_model, docs, embeddings, n_topics_final)

    # ------------------------------------------------------------------
    print("\n[STEP 10] Exporting results...")
    export(topic_model, df_results, topic_labels, eval_metrics, topics)

    # ------------------------------------------------------------------
    print("\n[STEP 11] Predicting on new documents...")
    new_texts = [
        "El congreso aprobó una nueva ley de reforma tributaria.",
        "El portero atajó el penal y el equipo avanzó a semifinales.",
        "La campaña de vacunación llegó a las zonas rurales más apartadas.",
        "Una empresa de software lanzó un modelo de lenguaje en español.",
        "Los incendios forestales arrasaron miles de hectáreas en el sur.",
    ]
    predict_new_documents(
        model_dir=OUTPUT_DIR / "bertopic_model",
        embedding_model=embedding_model,
        topic_labels=topic_labels,
        new_texts=new_texts,
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    n_noise = sum(1 for t in topics_final if t == -1)
    print(f"  Final topics         : {n_topics_final}")
    print(f"  Noise documents      : {n_noise} ({n_noise / len(topics_final) * 100:.1f} %)")
    print(f"  Coherence CV         : {eval_metrics['coherence_cv']:.4f}" if eval_metrics["coherence_cv"] else "  Coherence CV         : N/A")
    print(f"  Silhouette Score     : {eval_metrics['silhouette']:.4f}"   if eval_metrics["silhouette"]   else "  Silhouette Score     : N/A")
    print(f"  Output directory     : {OUTPUT_DIR}/")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)