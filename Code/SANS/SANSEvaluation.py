# -*- coding: utf-8 -*-
"""
Evaluating SANS on NER Test Data
"""

import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag
from sklearn_crfsuite import metrics
from nltk import sent_tokenize
import re
import joblib

def has_loc_suffix(word):
    """
    Identify if a given word has a suffix. SANS helper function.
    """
    suffix = ['pur', 'adi', 'iya', 'gaon', 'orest', 'ani', 'patti', 'palle', 'khurd', 'purwa', 'dih', 'chak', 'minor', 'garh', 'singh', 'uru', 'palem', 'ain', 'ganj', 'anga', 'and', 'padu', 'uzurg', 'utary', 'pet', 'attu', 'ane', 'angi', 'kh.', 'bk.'] #most common suffixes (top 30) obtained from suffix extractor
    for suf in suffix:
        if word.lower().endswith(suf) and word.lower() != suf:
            return(True)
    return(False)

def has_loc_domainwords(word, postagprev):
    """
    Identify if a given word has a domain word. SANS helper function.
    """
    domainwords = ['nagar','colony','street','road','hill','river','temple','village','sector', 'district', 'taluk', 'town', 'mutt', 'fort', 'masjid', 'church']
    for entry in domainwords:
        if word.lower() == entry:
            if postagprev in ['NNP','NNPS']:
                return(True)
    return(False)

def has_prep(word, postagnext):
    """
    Identify if a given word has a suffix. SANS helper function.
    """
    preps = ['near', 'via', 'in', 'from', 'between', 'at', 'versus', 'like', 'towards', 'of', 'toward', 'across'] # Place name prepositions with location likelihood scores greater than 0.1
    for prep in preps:
        if word.lower() == prep:
            if postagnext in ['NNP','NNPS']:
                return(True)
    return(False)

def get_wordshape(word):
    """
    Identify the shape of a given word. SANS helper function.
    """
    shape1 = re.sub('[A-Z]', 'X',word)
    shape2 = re.sub('[a-z]', 'x', shape1)
    return re.sub('[0-9]', 'd', shape2)

def is_in_gazetteer(word, postag, placeNames):
    """
    Identify if a given word is in the gazetteer. SANS helper function.
    """
    if postag in ['NNP', 'NNPS'] and word in placeNames:
        return True
    return False

def wordFeatures(sent, i, placeNames):
    """
    Generate features for sentences. SANS helper function.
    """
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.shape':get_wordshape(word),
        'hassuffix:':has_loc_suffix(word),
        'is_in_gazetteer:':is_in_gazetteer(word, postag, placeNames),
        'wordallcap': len([x for x in word if x.isupper()])==len(word),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:wordallcap': len([x for x in word1 if x.isupper()])==len(word1),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:prep': has_prep(word, postag1),
            '+1:hasdomain':has_loc_domainwords(word1,postag),
            '+1:wordallcap': len([x for x in word1 if x.isupper()])==len(word1),
        })
    else:
        features['EOS'] = True

    return features

def sentFeatures(sent, placeNames):
    """
    Given a sentence, generate features using the word2feature function for the sentence. SANS helper function.
    """
    return [wordFeatures(sent, i, placeNames) for i in range(len(sent))]

def sentLabels(sent):
    """
    Given a token, postag, and label associated with a sentence in training data, extract labels corresponding to each word. SANS helper function.
    """
    return [label for token, postag, label in sent]


def main():
    df_true = pd.read_csv("/Data/NERData/NERTestDataLabels.csv", encoding="latin-1") #contains gold standard labels for individual word (read the readme.txt file for details)
    df_eval = pd.read_csv("/Data/NERData//NERTestDataText.csv", encoding="latin-1") #contains the text i.e., news reports
    df_gazetteer = pd.read_csv('/Data/:Gazetteer/GazetteerIndia.csv', encoding='latin-1')
    placeNames = df_gazetteer['Placename'].values.tolist()
    
    SANS = joblib.load('/SALECode/Data/NER/crfNER895.pkl') #load the SANS
    
    # Preprocess the whole reports and store lists of sentences
    test_sentences = []
    for i, row in df_eval.iterrows():
        row_content = row['Content']
        row_content = re.sub(r"\.(?!\s|$)", ". ", row_content)
        row_content = re.sub('[^A-Za-z0-9.-:,!?\'\'\"\"()%]+', ' ', row_content)
        test_sentences.append([word_tokenize(sent) for sent in sent_tokenize(row_content)])
    
    
    #convert the lists of all sentences into a single list
    all_sentences = []
    for report in test_sentences:
            all_sentences.extend(report)
            
    #get the POS tags for all the words in the list containing all the sentences from the text data so as to input to CRF model
    sent_postags = []
    for sent in all_sentences:
        sent_postags.append(pos_tag(sent))
    
    # Get the features for each word in the sent_postags 
    X = [sentFeatures(s, placeNames) for s in sent_postags]
    
    y_SANS = SANS.predict(X) # labels predicted by SANS

    # Get the true labels from the NERTestDAtaLables file
    agg_func = lambda s: [t for w, t in zip(s["Word"].values.tolist(),
                                                               s["TruthNER"].values.tolist())]
    sent = df_true.groupby("Sentence").apply(agg_func) # get sentences from 
    y_true = [s for s in sent] # get the entity labels for each individual word in the ground truth (lables are in NERTestDataLabels file)
    
    
    labels = ['LOCATION', 'ORGANIZATION', 'PERSON', 'O'] # a word can have any of the these labels (i.e., a word can be any of the 4 entity types per ENAMEX entity classification scheme)

    # list of labels without 'O' i.e. 'LOCATION', 'ORGANIZATION', and, 'PERSON'. label 'O' was removed because we need to evaluate the CRF based model based on the three classes
    labels.remove('O')
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    
    # Evaluation using Precision, Recall, and F scores for SANS
    metrics.flat_classification_report(y_SANS, y_true, labels=sorted_labels, digits=3)
    
if __name__ == "__main__":
    main()  