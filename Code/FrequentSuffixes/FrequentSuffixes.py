# -*- coding: utf-8 -*-
"""
Construct a suffix tree using the place names from a gazetteer for a given geographic context and then mine the frequent suffixes from the tree
"""
import pandas as pd
import re
from functools import reduce  # forward compatibility for Python 3
import operator

def getChildDict(dictionary, keyList):
    """
    Given a list of keys of a dictionary, get the children of the keys in the list
    """
    return reduce(operator.getitem, keyList, dictionary)

def updateDict(dictionary, keyList, value):
    """
    Update dictionary.
    """
    getChildDict(dictionary, keyList[:-1])[keyList[-1]] = value

def getSupportConfidence(placeNames,suff):
    """
    compute the support and confidence score for a given suffix
    """
    if len(suff) < 3:
        return(0.1,0.6)
    prevsuff = suff[1:]
    support = len([place for place in placeNames if place.endswith(suff)])
    confidence = support/len([place for place in placeNames if place.endswith(prevsuff)])
    return(support/len(placeNames),confidence)

def createSuffixTree(placeNames):
    """
    Builds a suffix tree given a list of place names from a geographic context
    """
    suffixTree = {}
    for place in placeNames:
        revPlace = place[::-1] #since we are generating frequent suffixes, reverse the place name
        if len(suffixTree) == 0:
            for key in reversed(revPlace):
                suffixTree = {key: suffixTree}
        else:
            x = [] #variable to store list of keys of the suffix tree traversed or created
            temp = {} #temporary variable to store dictionary
            for key in revPlace:
                x.append(key)
                try:
                    temp = dict(getChildDict(suffixTree, x)) #get the child dictionary given the list of chars of the place names
                except:
                    updateDict(suffixTree, x, dict()) # add a child to the dictinary if no child dict exists
    return(suffixTree)

def generateFrequentSuffixes(suffixTree, suff, tempsuff, minSupp, minConf,frequentSuff, placeNames):
    """
    generate frequent suffix using the suffix tree
    """
    for key,val in suffixTree.items():
        suff = suff + key #add a child key to the parent key of the suffix tree i.e. dictionary
        if len(suffixTree.items()) > 1:
            tempsuff = suff[:-1] #whenever a key has two or more child keys in the suffix tree, store the suff for breadth first traversal
        if type(val) is dict:
            support,confidence = getSupportConfidence(placeNames, suff[::-1])
            if support > minSupp and confidence > minConf:
                frequentSuff = generateFrequentSuffixes(val, suff, tempsuff, minSupp, minConf,frequentSuff, placeNames)
            else:
                if len(frequentSuff) == 0 and len(suff[:-1]) > 2:    
                    frequentSuff.append(suff[:-1][::-1])
                else:
                    flag = False
                    for s in frequentSuff: #check if a suff already exits in the list of frequent suffixes
                        if s.endswith(suff[:-1][::-1]):
                            flag = True
                            break
                        if suff[:-1][::-1].endswith(s):
                            frequentSuff.remove(s)
                            break
                    
                    if flag == False and len(suff[:-1]) > 2:
                        frequentSuff.append(suff[:-1][::-1])
        suff = tempsuff
    return(frequentSuff)

def main():
    df_gazetteer = pd.read_csv('/Data/Gazetteer/GazetteerIndia.csv', encoding='latin-1')
    placeNames = df_gazetteer['Placename'].str.lower().tolist()
    placeNames = [place.strip() for place in placeNames]
    placeNames = [re.sub('\(*\)','', place) for place in placeNames] #remove braces from place names. The gazetteer for USA has some entries such as Campbell Settlement (historical). Remove '(historical)' from the name
    placeNames = [re.sub('[^A-Za-z0-9.]+', '', place) for place in placeNames]#suffix_nospecialcharacters
    
    minSupp = 0.00095 #Min value set so that there must must be at least minnumplaces number of places from a geographic context to be considered frequent 
    minConf = 0.5
    
    suff = '' #define these variables for generateFrequentSuffixes in main as it is a resursive function
    tempsuff = ''
    frequentSuff = []
    tempsuff = '' #temporary placeholder for suffix computation
    
    suffixTree = createSuffixTree(placeNames)
    frequentSuff = generateFrequentSuffixes(suffixTree, suff, tempsuff, minSupp, minConf, frequentSuff, placeNames)
    
if __name__ == "__main__":
    main() 
    