import datetime, json, os
import pandas as pd, numpy as np

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

from sklearn.feature_extraction.text import CountVectorizer


from src.config import modeling_config, paths
from src.commons import Logger

def postprocess(
    topic_model: BERTopic,
    docs: list[str],
    all_stopwords: list[str],
    n_topics_found: int,
    logger: Logger
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
    logger.info(f"  Reducing to ~{target} topics...")
    topic_model.reduce_topics(docs, nr_topics=target)
    topics_reduced = topic_model.topics_
    n_final = len(topic_model.get_topic_info().query("Topic != -1"))
    logger.info(f"  Topics after reduction: {n_final}")

    # --- 7.2  Build and assign readable labels ---
    topic_labels = {}
    for tid in topic_model.get_topics():
        if tid == -1:
            continue
        top3 = [w for w, _ in topic_model.get_topic(tid)[:3]]
        topic_labels[tid] = f"T{tid}: {' | '.join(top3)}"

    topic_model.set_topic_labels(topic_labels)
    logger.info(f"  Sample labels: {list(topic_labels.values())[:4]}")

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
    logger.info("  Topic representation updated.")

    return topics_reduced, topic_labels

def build_results_dataframe(
    docs: list[str],
    topics: list[int],
    topic_labels: dict[int, str]
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

def visualize(
    topic_model: BERTopic,
    docs: list[str],
    embeddings: np.ndarray,
    n_topics_final: int,
    logger: Logger
) -> None:
    """
    Generate five interactive Plotly charts and save each as a standalone
    HTML file in topic_modeling. Open any of them in a browser to explore.

    Charts produced:
        - document_map.html       : 2D scatter of every document colored by topic
        - topic_barchart.html     : top keywords per topic side by side
        - similarity_heatmap.html : pairwise cosine similarity between topics
        - topic_hierarchy.html    : dendrogram showing topic relationships
        - intertopic_distance.html: bubble chart where distance = dissimilarity
    """

    def _save(fig, name: str) -> None:
        path = os.path.join(paths['topic_modeling'], f"{name}.html")
        fig.write_html(str(path))
        logger.info(f"  Saved: {path}")

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

def export(
    topic_model: BERTopic,
    df_results: pd.DataFrame,
    topic_labels: dict[int, str],
    eval_metrics: dict,
    topics_initial: list[int],
    logger: Logger
) -> None:
    """
    Write all results to topic_modeling:

        documents_with_topics.csv  : one row per document with topic + label
        topic_info.csv             : BERTopic's built-in topic metadata table
        topic_keywords.json        : keywords and scores for every topic
        evaluation_metrics.csv     : coherence, silhouette, and run metadata
        bertopic_model/            : serialized model (safetensors format)
    """

    # Documents CSV
    path = os.path.join(paths['topic_modeling'], "documents_with_topics.csv") #topic_modeling / "documents_with_topics.csv"
    df_results.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  Documents CSV      : {path}")

    # Topic info CSV
    path = os.path.join(paths['topic_modeling'], "topic_info.csv") #topic_modeling / "topic_info.csv"
    topic_model.get_topic_info().to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  Topic info CSV     : {path}")

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
    path = os.path.join(paths['topic_modeling'], "topic_keywords.json") #topic_modeling / "topic_keywords.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keywords_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"  Keywords JSON      : {path}")

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
        "embedding_model"  : modeling_config['EMBEDDING_MODEL_NAME'],
        "timestamp"        : datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }])
    path = os.path.join(paths['topic_modeling'], "evaluation_metrics.csv") #topic_modeling / "evaluation_metrics.csv"
    df_eval.to_csv(path, index=False)
    logger.info(f"  Evaluation CSV     : {path}")

    # Serialized model
    path = os.path.join(paths['topic_modeling'], "bertopic_model") #topic_modeling / "bertopic_model"
    topic_model.save(
        str(path),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,   # embedding model is reloaded by name on load
    )
    logger.info(f"  Model saved        : {path}")