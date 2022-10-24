# -*- coding: utf-8 -*-
"""
Get most frequent suffixes for India.
"""

import pandas as pd
import re
from time import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df_gazetteer = pd.read_csv('/Data/Gazetteers/GazetteerIndia.csv', encoding='latin-1')
area_names = df_gazetteer['Placename'].str.lower().tolist()


minnumplaces = 1000 #Min value set so that there must must be at least minnumplaces number of places from a country to be considered frequent 
suffix_2 = []
suffix_3 = []
suffix_4 = []
suffix_5 = []

for names in area_names:
    names = str(names).strip()
    names = re.sub('\(*\)','', names).strip() #remove braces from place names. The gazetteer for USA has some entries such as Campbell Settlement (historical). Remove '(historical)' from the name
    names = re.sub('[^A-Za-z0-9]+', '', names) #suffix_nospecialcharacters

    try:
        #extract suffixes of length 2-5 from place names and store in arrays
        if len(names) > 1:
            suffix_2.append(names[-2:])
        if len(names) > 2:
            suffix_3.append(names[-3:])
        if len(names) > 3:
            suffix_4.append(names[-4:])
        if len(names) > 4:
            suffix_5.append(names[-5:])
    except:
        continue

# Count the number of place names with suffixes of length 2-5
t = time()      
suffix_2_set = list(set(suffix_2))
count_suffix_2 = [suffix_2.count(suff) for suff in suffix_2_set]
print('Time to count suffix 2: {} mins'.format(round((time() - t) / 60, 2)))

t = time() 
suffix_3_set = list(set(suffix_3))
count_suffix_3 = [suffix_3.count(suff) for suff in suffix_3_set]
print('Time to count suffix 3: {} mins'.format(round((time() - t) / 60, 2)))


t = time() 
suffix_4_set = list(set(suffix_4))
count_suffix_4 = [suffix_4.count(suff) for suff in suffix_4_set]
print('Time to count suffix 4: {} mins'.format(round((time() - t) / 60, 2)))


t = time() 
suffix_5_set = list(set(suffix_5))
count_suffix_5 = [suffix_5.count(suff) for suff in suffix_5_set]
print('Time to count suffix 5: {} mins'.format(round((time() - t) / 60, 2)))


common_suffixes = []
common_suffixes_count = []

t = time() 
#Group common suffixes along with their counts e.g. ['ville','vill','vil','vi'] and their counts
for i in range(0, len(suffix_5_set)):
    for j in range(0, len(suffix_4_set)):
        if suffix_5_set[i].endswith(suffix_4_set[j]):
            for k in range(0, len(suffix_3_set)):
                if suffix_4_set[j].endswith(suffix_3_set[k]):
                    for l in range(0, len(suffix_2_set)):
                        if suffix_3_set[k].endswith(suffix_2_set[l]):
                            #print(str(i) + ',' + str(j) + ',' + str(k) +','+str(l))
                            common_suffixes.append([suffix_5_set[i], suffix_4_set[j], suffix_3_set[k], suffix_2_set[l]])
                            common_suffixes_count.append([count_suffix_5[i], count_suffix_4[j], count_suffix_3[k], count_suffix_2[l]])
                            break
                    break
            break
print('Time to find common suffs: {} mins'.format(round((time() - t) / 60, 2)))

#Compute the MLE using the bigram formulation for each suffix in the group
bigram_MLE = []
for i in range(0, len(common_suffixes)):
    bigram_MLE.append( [common_suffixes_count[i][0]/common_suffixes_count[i][1], common_suffixes_count[i][1]/common_suffixes_count[i][2], common_suffixes_count[i][2]/common_suffixes_count[i][3], 0])

final_suffixes = []
final_suffixes_count = []
for i in range(0, (len(common_suffixes))):
    if common_suffixes_count[i][2] > 0:
        if bigram_MLE[i][2] > 0.5 and common_suffixes_count[i][2] > 1000:
            if bigram_MLE[i][1] > 0.5 and common_suffixes_count[i][1] > 1000:
                if bigram_MLE[i][0] > 0.5 and common_suffixes_count[i][0] > 1000:
                    final_suffixes.append(common_suffixes[i][0])
                    final_suffixes_count.append(common_suffixes_count[i][0])
                else:
                    final_suffixes.append(common_suffixes[i][1])
                    final_suffixes_count.append(common_suffixes_count[i][1])    
            else:
                final_suffixes.append(common_suffixes[i][2])
                final_suffixes_count.append(common_suffixes_count[i][2])
                


df_suffix = pd.DataFrame(columns = ['Suffix', 'Count'])
for i in range(0, len(final_suffixes)):
    df=pd.DataFrame([[final_suffixes[i], final_suffixes_count[i]]], columns=df_suffix.columns)
    df_suffix = df_suffix.append(df)


df_suffix = df_suffix.drop_duplicates()

df_suffix = df_suffix.sort_values( by=['Count'], ascending=False)

# Remove parent if child is already present in the final list i.e. if 'aon' and 'gaon' are both selected. Then keep 'gaon' only for the suffix.
final_suffixes = list(set(final_suffixes))
len_3 = [suff for suff in final_suffixes if len(suff) == 3]
len_4 = [suff for suff in final_suffixes if len(suff) == 4]
len_5 = [suff for suff in final_suffixes if len(suff) == 5]
repeated_suffix_list = []

for i in range(0, len(len_3)):
    for j in range(0, len(len_4)):
        if len_4[j].endswith(len_3[i]):
            repeated_suffix_list.append(len_3[i])

for i in range(0, len(len_3)):
    for j in range(0, len(len_5)):
        if len_5[j].endswith(len_3[i]):
            repeated_suffix_list.append(len_3[i])

for i in range(0, len(len_4)):
    for j in range(0, len(len_5)):
        if len_5[j].endswith(len_4[i]):
            repeated_suffix_list.append(len_4[i])          

suffix_list = [suff for suff in final_suffixes if suff not in repeated_suffix_list]


df_suffix_final = pd.DataFrame(columns=['Suffix', 'Count'])

for i, row in df_suffix.iterrows():
    if row['Suffix'] in suffix_list and row['Count'] > minnumplaces:
        df = pd.DataFrame([[row['Suffix'], row['Count']]], columns = df_suffix_final.columns)
        df_suffix_final = df_suffix_final.append(df)


dict_suff = {}
for i, row in df_suffix_final[0:8].iterrows():
    dict_suff[row['Suffix']] = row['Count']
    
wordcloud = WordCloud(background_color="lightyellow")
wordcloud.generate_from_frequencies(frequencies=dict_suff)
plt.figure(figsize=[7,7])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()