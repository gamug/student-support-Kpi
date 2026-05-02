"""
The script contains the functions to process the documents breaking the answer into sentences
and analyzing the sentiment of the sentences
"""

import spacy
import pandas as pd
import transformers

def sentencizer(answers: pd.DataFrame, nlp: spacy.language.Language) -> pd.DataFrame:
    """Splits the answers into sentences and creates a DataFrame with the following columns:
    - student: the name of the student
    - question: the name of the question
    - sentence_number: the number of the sentence
    - sentence: the text of the sentence
    
    Args:
        answers (pd.DataFrame): the DataFrame with the answers of the students
        nlp: the spacy model
    
    Returns:
        pd.DataFrame: the DataFrame with the sentences of the answers
    """
    corpus = []
    for student in answers.index:
        for question in answers.columns:
            doc = nlp(answers[question].loc[student])
            sentences = [sent.text.replace('\n', '') for sent in doc.sents]
            df = pd.DataFrame({
                'student': student,
                'question': question,
                'sentence_number': list(range(len(sentences))),
                'sentence': sentences
            })
            corpus.append(df)
    return pd.concat(corpus).reset_index(drop=True)

def sentiment_analyzer(corpus: pd.DataFrame, sentiment_model: transformers.pipelines.base.Pipeline) -> pd.DataFrame:
    """Analyzes the sentiment of the sentences and adds it to the DataFrame with the following columns:
    - sentiment_label: the label of the sentiment (positive, negative, neutral)
    - sentiment_score: the score of the sentiment
    
    Args:
        corpus (pd.DataFrame): the DataFrame with the sentences of the answers. This dataframe is the output of the sentencizer function
        sentiment_model: the sentiment model to predict the sentiment in the corpus sentences
        
    Returns:
        pd.DataFrame: the DataFrame with the sentiments of the sentences
    """
    sentiment = sentiment_model(corpus.sentence.tolist())
    sentiment = pd.DataFrame(sentiment).rename(
        {'label': 'sentiment_label', 'score': 'sentiment_score'},
        axis=1
    )
    return pd.concat([corpus, sentiment], axis=1)