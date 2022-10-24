# -*- coding: utf-8 -*-
"""
Spatially-aware location extraction algorithm
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.tokenize import MWETokenizer
mwetokenizer = MWETokenizer([('New','Delhi')], separator=' ') #inlcuding the format for multi word tokenizer. This is useful for Pathway 2 i.e., extracting place names using a Gaz and POS
import joblib
import re

def binary_search(arr, low, high, x): 
  
    # Check base case 
    if high >= low: 
  
        mid = (high + low) // 2
        if arr[mid] == x: 
            return True 
  
        # If element is smaller than mid, then it can only 
        elif arr[mid] > x: 
            return binary_search(arr, low, mid - 1, x) 
  
        # Else the element can only be present in right subarray 
        else: 
            return binary_search(arr, mid + 1, high, x) 
  
    else: 
        # Element is not present in the array 
        return False
#Function for SANS

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


# Functions to extract place names using SANS in the test dataset.
def get_locs_NER(doc, NER, placeNames):
    output = get_NER_5WNER(sent_tokenize(doc), NER, placeNames) #sentence tokenize the document and pass it as parameter to get_NER_5WNER to get the NER tags
    locations = concat_placenames(output)
    
    list_set = set(locations) 
    
    # convert the set to the list 
    unique_locs = (list(list_set))
    return unique_locs

def concat_placenames(original_tags):
    """
    Combine names of the locations if the locations consists of two or more words eg. New Delhi, New York, etc.
    """
    locations = []
    l = len(original_tags)
    i=0;
    # Iterate over the tagged words.
    while i<l:
        e,t = original_tags[i]
        # If it's a location, then check the next words.
        if t == 'LOCATION':
            j = 1
            s = e
            # Verify the tags for the next words.
            while i+j<len(original_tags):
                # If the next words are also locations, then concatenate them to make a longer string. This is useful for place names with multiple words. e.g., New Delhi
                if original_tags[i+j][1] == 'LOCATION':
                    s = s+" "+original_tags[i+j][0]
                    j+=1
                else:
                    break
            i = i+j
            # Save the locations to a locations list
            locations+=[s]
        else:
            i=i+1
            
    return locations


def get_NER_5WNER(doc, NER, placeNames):
    """
    Use SANS to identify location names in text.
    """
    input_ner = []
    tags = []
    for sent in doc:
        text_tokens = pos_tag(word_tokenize(sent))
        input_ner.append(sentFeatures(text_tokens,placeNames)) #create input to the 5WNER tagger from the text file
        tags.extend(word_tokenize(sent))
    output_ner = (NER.predict(input_ner))
    ner_list = [item for sublist in output_ner for item in sublist] #convert the list of list i.e. output_ner which contains all the NER tags for each sentence of the report
    ner_tags = [(w,t) for w,t in zip(tags, ner_list)] #give output same as that of Stanford NER [(word, NER tag)]
    return(ner_tags)

def getLocationsGazPOSandSAPR(reports, preps, placeName, suffix):
    "Get all the locations mentioned in text using Gazetteer and POS"
    tokenizedContent = mwetokenizer.tokenize(word_tokenize(reports))
    tokenized_tags = pos_tag(tokenizedContent)
    locationgaz = []
    locationsapr = []
    for j in range(0,len(tokenized_tags)):
        if tokenized_tags[j][1] in ['NNP', 'NNPS']:
            if binary_search(placeName, 0, len(placeName), tokenized_tags[j][0]) == True:
                locationgaz.append(tokenized_tags[j][0].title())
            else:
                
                if tokenized_tags[j-1][0].lower() in preps: 
                    place = tokenized_tags[j][0].title()
                    k = j + 1
                    while k < len(tokenized_tags): #Get all the NNPs before a prep. They may represent a multi-word place names
                        if tokenized_tags[k][1] == 'NNP' or tokenized_tags[k][1] == 'NNPS':
                            place = place + ' ' + tokenized_tags[k][0].title()
                            k = k + 1
                        else: 
                            break
                    locationsapr.append(place)
                
                for suff in suffix:
                    if tokenized_tags[j][0].lower().endswith(suff):
                        locationsapr.append(tokenized_tags[j][0].title())
    
    return(locationgaz, locationsapr)
        

def main():
    df_groundtruth = pd.read_csv('/Data/SALEGroundTruth/GroundTruth.csv', encoding='latin-1')

    df_gazetteer = pd.read_csv('/Data/Gazetteer/EnhancedHierarchicalGazetteer.csv', encoding='latin-1')
    
    placeName = df_gazetteer['Placename'].values.tolist()
    placeName.sort()
    
    # load SANS from the saved file
    SANS = joblib.load('/Data/NERData/crfNER895.pkl')
    
    #Using multi word tokenizer to tokenize states with two words so that they can be compared easily
    for place in placeName:
        if len(word_tokenize(place)) == 2:
            mwetokenizer.add_mwe((word_tokenize(place)[0], word_tokenize(place)[1]))
        if len(word_tokenize(place)) == 3:
            mwetokenizer.add_mwe((word_tokenize(place)[0], word_tokenize(place)[1], word_tokenize(place)[2])) 
        if len(word_tokenize(place)) == 4:
            mwetokenizer.add_mwe((word_tokenize(place)[0], word_tokenize(place)[1], word_tokenize(place)[2], word_tokenize(place)[3]))
        if len(word_tokenize(place)) == 5:
            mwetokenizer.add_mwe((word_tokenize(place)[0], word_tokenize(place)[1], word_tokenize(place)[2], word_tokenize(place)[3], word_tokenize(place)[4]))
        if len(word_tokenize(place)) == 6:
            mwetokenizer.add_mwe((word_tokenize(place)[0], word_tokenize(place)[1], word_tokenize(place)[2], word_tokenize(place)[3], word_tokenize(place)[4], word_tokenize(place)[5]))
        if len(word_tokenize(place)) == 7:
            mwetokenizer.add_mwe((word_tokenize(place)[0], word_tokenize(place)[1], word_tokenize(place)[2], word_tokenize(place)[3], word_tokenize(place)[4], word_tokenize(place)[5], word_tokenize(place)[6]))
        if len(word_tokenize(place)) == 8:
            mwetokenizer.add_mwe((word_tokenize(place)[0], word_tokenize(place)[1], word_tokenize(place)[2], word_tokenize(place)[3], word_tokenize(place)[4], word_tokenize(place)[5], word_tokenize(place)[6], word_tokenize(place)[7]))
        if len(word_tokenize(place)) == 9:
            mwetokenizer.add_mwe((word_tokenize(place)[0], word_tokenize(place)[1], word_tokenize(place)[2], word_tokenize(place)[3], word_tokenize(place)[4], word_tokenize(place)[5], word_tokenize(place)[6], word_tokenize(place)[7], word_tokenize(place)[8]))
    
    
    locationGaz = [] # locations identified using Pathway 2
    locationSAPR = [] # location identified using Pathway 3
    preps = ['near', 'via', 'in', 'from', 'between', 'at', 'versus', 'like', 'towards', 'of', 'toward', 'across']
    suffix = ['pur', 'adi', 'iya', 'gaon', 'orest', 'ani', 'patti', 'palle', 'khurd', 'purwa', 'dih', 'chak', 'minor', 'garh', 'singh', 'uru', 'palem', 'ain', 'ganj', 'anga', 'and', 'padu', 'uzurg', 'utary', 'pet', 'attu', 'ane', 'angi', 'kh.', 'bk.']
    
    for i, row in df_groundtruth.iterrows():
        row_title_content = row['Title'] + '. ' + row['Content']
        locationgaz, locationsapr = getLocationsGazPOSandSAPR(row_title_content, preps, placeName, suffix) #locations identified using Pathway 2 (high confidence) and pathway 3 (low confidence)                  
        locationGaz.append(locationgaz)
        locationSAPR.append(locationsapr)
    
    # locations identified using Pathway 1
    locationSAN = []     
    for i, row in df_groundtruth.iterrows():
        row_title_content = row['Title'] + '. ' + row['Content']
        locationSAN.append(get_locs_NER(row_title_content, SANS, placeName))
    
    locationSALE = [] #all locations identified by 3 pathways combined. Since we focus on identifying all the locations mentioned, irrespective of the confidence values associated with the locations, in this research, we do not use the confidence values here. The confidence is useful for extension of this work
    for i in range(0, len(df_groundtruth)):
        location = []
        location.extend(locationSAN[i])
        location.extend(locationGaz[i])
        location.extend(locationSAPR[i])
        location = list(set(location))
        locationSALE.append(location)
        
        
if __name__ == "__main__":
    main()  