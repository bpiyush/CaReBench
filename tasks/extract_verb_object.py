"""Use Spacy to annotate the verb and object of a sentence."""
import os
import sys
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import faiss
import os
from pathlib import Path
from tqdm import tqdm
import gc
import pandas as pd
from pyarrow import csv as pcsv
import decord
import spacy
import re
from typing import Tuple, Optional

import shared.utils as su


def clean_sentence(sentence: str) -> str:
    """Clean and normalize the input sentence."""
    # Remove #C, #O, #c markers and person identifiers
    cleaned = re.sub(r'#[CcOo]\s*', '', sentence)
    
    # Only remove the subject "C " but keep other words
    cleaned = re.sub(r'\bC\s+', '', cleaned)
    
    # Clean up extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def extract_verb_object_debug(sentence: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the main action verb and its primary object with debugging.
    """
    cleaned_sentence = clean_sentence(sentence)
    print(f"  Cleaned: '{cleaned_sentence}'")
    
    if not cleaned_sentence:
        return (None, None)
    
    doc = nlp(cleaned_sentence)
    
    # Debug: print all tokens and their POS tags
    print(f"  Tokens: {[(token.text, token.pos_, token.dep_) for token in doc]}")
    
    # Find ANY verb first
    verbs = [token for token in doc if token.pos_ == "VERB"]
    print(f"  Verbs found: {[(v.text, v.lemma_) for v in verbs]}")
    
    if not verbs:
        return (None, None)
    
    # Take the first meaningful verb
    main_verb = verbs[0]
    verb_lemma = main_verb.lemma_
    
    print(f"  Main verb: '{main_verb.text}' (lemma: '{verb_lemma}')")
    print(f"  Verb children: {[(child.text, child.dep_) for child in main_verb.children]}")
    
    # Strategy 1: Look for direct object
    for child in main_verb.children:
        if child.dep_ == "dobj":
            print(f"  Found dobj: '{child.text}'")
            return (verb_lemma, child.text)
    
    # Strategy 2: Look for prepositional objects
    for child in main_verb.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    print(f"  Found pobj: '{grandchild.text}'")
                    return (verb_lemma, grandchild.text)
    
    print(f"  No object found")
    return (verb_lemma, None)

def extract_verb_object(sentence: str, nlp: spacy.Language) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the main action verb and its primary object (non-debug version).
    """
    cleaned_sentence = clean_sentence(sentence)
    
    if not cleaned_sentence:
        return (None, None)
    
    doc = nlp(cleaned_sentence)
    
    # Find the ROOT token (main verb) - even if spaCy tags it wrong
    main_verb = None
    for token in doc:
        if token.dep_ == "ROOT":
            main_verb = token
            break
    
    if not main_verb:
        return (None, None)
    
    # Get the lemma - this will give us the base form even if POS is wrong
    verb_lemma = main_verb.lemma_
    
    # Strategy 1: Look for direct object or appositive (like "fixes wires")
    for child in main_verb.children:
        if child.dep_ in ["dobj", "appos"]:
            return (verb_lemma, child.text)
    
    # Strategy 2: Look for prepositional objects
    for child in main_verb.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return (verb_lemma, grandchild.text)
    
    return (verb_lemma, None)

# Test with a few key samples
def test_samples():
    """Test with a handful of problematic samples."""
    test_sentences = [
        "#C C stares at the clothes on the hangers",
        "#c c walks to the socket", 
        "#C C dips the paint brush in the paint can",
        "#C C fixes wires",
        "#C C talks to lady Z"
    ]
    
    print("Testing fixed function:")
    print("-" * 50)
    
    for sentence in test_sentences:
        verb, obj = extract_verb_object(sentence)
        print(f"'{sentence}'")
        print(f"  -> Verb: '{verb}', Object: '{obj}'")
        print()



if __name__ == "__main__":
    
    model_id = "en_core_web_sm"
    # model_id = "en_core_web_trf"
    use_gpu = False

    if use_gpu:
        print("Enabling spaCy GPU support")
        spacy.require_gpu()

    # Load data
    data_dir = "/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap"
    
    # Configure the input CSV and output CSV paths
    # filename = "clips_duration_1s_to_10s"
    filename = "ego4d_5.3M_clips"
    csv_path = f"{data_dir}/{filename}.csv"
    save_path = f"{data_dir}/{filename}_verb_object_{model_id}.csv"

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print("Number of rows in df:", len(df))
    print("." * 100)

    # Load the English model
    print(f"Loading English model ({model_id})")
    nlp = spacy.load(model_id)
    
    # Debug
    test_samples()
    print("." * 100)

    # Run on captions
    captions = df.caption.tolist()
    iterator = su.log.tqdm_iterator(captions, desc="Running inference on sample")
    outputs = {"verb": [], "object": [], "caption": []}
    for c in iterator:
        try:
            # v, o = extract_simple_action_object(c)
            v, o = extract_verb_object(c, nlp)
        except:
            v, o = None, None
        outputs['verb'].append(v)
        outputs['object'].append(o)
        outputs['caption'].append(c)
    outputs = pd.DataFrame(outputs)
    
    # Save path
    print(f"Saving to {save_path}")
    outputs.to_csv(save_path, index=False)
    print("." * 100)

