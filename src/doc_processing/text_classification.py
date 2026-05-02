"""
This script classifies text as either "STUDENT" or "PROGRAM".
"""

import spacy
from collections import defaultdict
from typing import List, Dict, Optional

from src.config import program_keywords, first_person_pronouns

def classify_sentence(
    sentence: str,
    nlp: spacy.language.Language,
) -> Dict:
    """
    Classify if a sentence is about the student (self) or the program.
    
    Args:
        sentence: Individual sentence to classify
        program_keywords: List of program-specific terms
    
    Returns:
        Classification result with scores and evidence
    """
    doc = nlp(sentence)
    
    scores = {
        'student': 0,
        'program': 0
    }
    
    evidence = defaultdict(list)
    
    # Track if sentence has mixed content
    has_student_markers = False
    has_program_markers = False
    
    for token in doc:
        # STUDENT MARKERS: First person pronouns and possessives
        if token.pos_ in ["PRON", "DET"] and token.text.lower() in first_person_pronouns:
            scores['student'] += 2
            evidence['student'].append(f"1st_person_pronoun: '{token.text}'")
            has_student_markers = True
        
        # STUDENT MARKERS: First person verbs
        if token.pos_ in ["VERB", "AUX"]:
            morph = token.morph.to_dict()
            if morph.get('Person') == '1':
                scores['student'] += 1
                evidence['student'].append(f"1st_person_verb: '{token.lemma_}'")
                has_student_markers = True
        
        # PROGRAM MARKERS: Program-related keywords
        lemma_lower = token.lemma_.lower()
        text_lower = token.text.lower()
        
        if lemma_lower in program_keywords or text_lower in program_keywords:
            scores['program'] += 3
            evidence['program'].append(f"keyword: '{token.text}'")
            has_program_markers = True
        
        # PROGRAM MARKERS: Third person subjects (especially nouns)
        if token.dep_ == "nsubj":
            morph = token.morph.to_dict()
            # Third person subjects
            if (morph.get('Person') == '3' or token.pos_ in ["NOUN", "PROPN"]) and (lemma_lower in program_keywords or text_lower in program_keywords):
                scores['program'] += 2
                evidence['program'].append(f"program_subject: '{token.text}'")
                has_program_markers = True
    
    # Determine classification
    if scores['student'] > scores['program']:
        classification = "STUDENT"
    elif scores['program'] > scores['student']:
        classification = "PROGRAM"
    elif scores['student'] == scores['program'] and scores['student'] > 0:
        classification = "MIXED"
    else:
        classification = "UNCLEAR"
    
    # Flag mixed sentences (both student and program elements)
    is_mixed = has_student_markers and has_program_markers
    
    return {
        'classification': classification,
        'scores': scores,
        'evidence': dict(evidence),
        'is_mixed': is_mixed,
        'sentence': sentence.strip()
    }


def classify_sentences_batch(
    sentences: List[str],
    nlp: spacy.language.Language
) -> List[Dict]:
    """
    Classify a list of pre-segmented sentences.
    
    Args:
        sentences: List of individual sentences
        program_keywords: Optional custom program keywords
    
    Returns:
        List of classification results
    """
    results = []
    
    for sentence in sentences:
        if sentence.strip():  # Skip empty sentences
            result = classify_sentence(sentence, nlp)
            results.append(result)
    
    return results