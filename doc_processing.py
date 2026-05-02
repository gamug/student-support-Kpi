import argparse, os
import pandas as pd
from typing import Any
from src.doc_processing import classify_sentences_batch, sentencizer, sentiment_analyzer
from src import nlp, sentiment_model
from src.commons import Logger

logger = Logger("Data Processing Module")

def main(answers: pd.DataFrame) -> Any:
    try:
        logger.info('Breaking answers into sentences...')
        corpus = sentencizer(answers, nlp)
        logger.info('Performing sentiment analysis...')
        1/0
        corpus = sentiment_analyzer(corpus, sentiment_model)
        logger.info('Classifying sentences into PROGRAM and STUDENT...')
        results = pd.DataFrame(classify_sentences_batch(corpus.sentence.tolist(), nlp))
        results.to_excel(os.path.join('..', 'output', 'classification_results.xlsx'), index=False)
        corpus = corpus.assign(sentence_subject=results.classification)
    except Exception as e:
        logger.error(e)
        logger.info('Data processing FAILS')
        return None
    return corpus

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Document processing pipeline")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the student answers",
        default=os.path.join('..', 'questions', 'dummy-data.xlsx')
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to put processing results",
        default=os.path.join('..', 'output', 'stundent_support_corpus.xlsx')
    )
    args = parser.parse_args()

    answers = pd.read_excel(
        args.input_path,
        index_col=0,
        sheet_name='Questionario'
    )
    if corpus := main(answers):
        logger.info('Saving results...')
        corpus.to_excel(args.output_path, index=False)
        logger.info('Data processing was SUCCESFULLY completed')