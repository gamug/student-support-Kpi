import datetime, warnings
import numpy as np
import plotly.io as pio
warnings.filterwarnings("ignore")

from src.topic_modeling.document_preparation import load_data, build_stopwords, preprocess, compute_embeddings
from src.topic_modeling.model_tools import build_topic_model, train, evaluate
from src.topic_modeling.result_building import postprocess, build_results_dataframe, visualize, export
from src.config import modeling_config, paths
from src.commons import Logger

logger = Logger("Topic Modeling Module")


def main():
    pio.renderers.default = "browser"
    np.random.seed(modeling_config['SEED'])

    logger.info("=" * 60)
    logger.info("TOPIC MODELING IN SPANISH WITH BERTOPIC")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 1] Loading data...")
    docs_raw = load_data()
    logger.info(f"  Raw documents loaded: {len(docs_raw)}")

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 2] Preprocessing...")
    all_stopwords = build_stopwords()
    docs          = preprocess(docs_raw, logger)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 3] Computing embeddings...")
    embedding_model, embeddings = compute_embeddings(docs, logger)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 4] Building BERTopic model...")
    topic_model = build_topic_model(all_stopwords)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 5] Training...")
    topics, probs = train(topic_model, docs, embeddings, logger)
    n_topics_found = len(topic_model.get_topic_info().query("Topic != -1"))

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 6] Evaluating...")
    eval_metrics = evaluate(topic_model, docs, embeddings, topics, logger)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 7] Post-processing...")
    topics_final, topic_labels = postprocess(
        topic_model, docs, all_stopwords, n_topics_found, logger
    )
    n_topics_final = len(topic_model.get_topic_info().query("Topic != -1"))

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 8] Building results DataFrame...")
    df_results = build_results_dataframe(docs, topics_final, topic_labels)
    logger.info(f"  Topic labels: {', '.join(list(df_results.topic_label.unique()))}")

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 9] Generating visualizations...")
    visualize(topic_model, docs, embeddings, n_topics_final, logger)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[STEP 10] Exporting results...")
    export(topic_model, df_results, topic_labels, eval_metrics, topics, logger)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    n_noise = sum(1 for t in topics_final if t == -1)
    logger.info(f"  Final topics         : {n_topics_final}")
    logger.info(f"  Noise documents      : {n_noise} ({n_noise / len(topics_final) * 100:.1f} %)")
    logger.info(f"  Coherence CV         : {eval_metrics['coherence_cv']:.4f}" if eval_metrics["coherence_cv"] else "  Coherence CV         : N/A")
    logger.info(f"  Silhouette Score     : {eval_metrics['silhouette']:.4f}"   if eval_metrics["silhouette"]   else "  Silhouette Score     : N/A")
    logger.info(f"  Output directory     : {paths['topic_modeling']}") #f"  Output directory     : {OUTPUT_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()