# -*- coding: utf-8 -*-
"""
Spatially-aware Named Entity Recognition System Training
"""

import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
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
    df_data = pd.read_csv("/Data/NERData/NERTrainingData.csv", encoding="latin-1")
    df_gazetteer = pd.read_csv('/Data/Gazetteer/GazetteerIndia.csv', encoding='latin-1')
    placeNames = df_gazetteer['Placename'].values.tolist()
    
    agg_func = lambda s: [(w, p, n) for w, p, n in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["NER"].values.tolist())]
    sent = df_data.groupby("Sentence").apply(agg_func) # get sentences from  
    sentences = [s for s in sent] # Create a list of all the words and sentences from the data
    
    # Get the features and labels for each word in the training data
    X = [sentFeatures(s, placeNames) for s in sentences]
    y = [sentLabels(s) for s in sentences]
    
    # define fixed parameters and parameters to search and train the CRF on training data
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=50,
        all_possible_states = True,
        linesearch='Backtracking',
        max_linesearch=30,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }
    
    crf.fit(X, y)

    labels = list(crf.classes_) #all the labels 'LOCATION', 'ORGANIZATION', 'PERSON' and 'O'
    
    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=list(labels))
    
    
    # search for best c1 (l1) and c2 (l2) in the param space with the best f1_score. This helps to tune the parameters of the CRF for different features and determine the weight for features included in the CRF. We want to weight higher those words that represent locations.
    rs = RandomizedSearchCV(crf, params_space,
                            cv=10,
                            verbose=5,
                            n_jobs=-2,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X, y) 
    
    crf = rs.best_estimator_  #assign the best parameters to the crf
    
    joblib_file = "/SALECode/Data/NER/crfNER895.pkl"  #store the CRF as a pickle file so that it can be imported and used.
    joblib.dump(crf, joblib_file)
    
if __name__ == "__main__":
    main()  