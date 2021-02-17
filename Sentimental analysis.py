#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
import nltk
import re
from collections import Counter

train = pd.read_csv('drugsComTrain_raw.csv', parse_dates=True) #index_col=0 - remove index column, parse_date - to normalize date type
test = pd.read_csv('drugsComTest_raw.csv', parse_dates=True)


# # Training dataset
train = train.dropna()
train = train[~train.condition.str.contains("</span>")]
train = train[~train.condition.str.contains("Not Listed / Othe")]
train
#159959 rows × 7 columns

import time
start_time=time.time()

# NLP
train['lower'] = train["review"].str.lower()

def identify_tokens(row):
    stories_lower = row.loc['lower']
    tokens = nltk.word_tokenize(stories_lower)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return (token_words)

train.loc[:,'tokens'] = train.apply(identify_tokens, axis=1)

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
stopWords = list(stopWords)
no_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# stops = list(zip(stopWords, no_list))
for i in no_list : 
    stopWords.append(i) 
    
def remove_stops(row):
    my_list = row['tokens']
    meaningful_words = [w for w in my_list if not w in stopWords]
    return (meaningful_words)

train.loc[:,'meaningful_words'] = train.apply(remove_stops, axis=1)

def remove_condition(row):
    my_list = row['meaningful_words']
    x =  'rosacea'
    meaningful_words_xcond = [w for w in my_list if w != x]
    return (meaningful_words_xcond)

train.loc[:,'meaningful_words_xcond'] = train.apply(remove_condition, axis=1)

def rejoin_words(row):
    my_list = row['meaningful_words_xcond']
    joined_words = ( " ".join(my_list))
    return (joined_words)


train.loc[:,'joined_words'] = train.apply(rejoin_words, axis=1)

end_time=time.time()
print("總共執行了:  %d min  %d sec" % (int((end_time-start_time)/60),(end_time-start_time)%60))


rating_df = train.loc[:,['uniqueID', 'rating']]
ratings = rating_df.rating.values

y = []
for i in ratings:
    i = float(i)
    y.append(i)

ratings = list(y)
rating_df['rating_1'] = ratings

rating_df['Label'] = rating_df['rating_1'].apply(lambda x: 1 if x >=5 else 0)
rating_df.Label.value_counts()

rating_df = rating_df.drop(columns = ['rating'])
train_data = train.merge(rating_df, on = 'uniqueID', how='left')

# # Testing dataset
test_data = test.dropna()
mask = test_data.condition.str.contains("</span>")
test_data = test_data[~mask]

mask1 = test_data.condition.str.contains("Not Listed / Othe")
test_data = test_data[~mask1]

# NLP
import time
start_time=time.time()

test_data['lower'] = test_data["review"].str.lower()

def identify_tokens(row):
    stories_lower = row.loc['lower']
    tokens = nltk.word_tokenize(stories_lower)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return (token_words)

test_data.loc[:,'tokens'] = test_data.apply(identify_tokens, axis=1)

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
stopWords = list(stopWords)
no_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# stops = list(zip(stopWords, no_list))
for i in no_list : 
    stopWords.append(i) 
    
def remove_stops(row):
    my_list = row['tokens']
    meaningful_words = [w for w in my_list if not w in stopWords]
    return (meaningful_words)

test_data.loc[:,'meaningful_words'] = test_data.apply(remove_stops, axis=1)

def remove_condition(row):
    my_list = row['meaningful_words']
    x =  'rosacea'
    meaningful_words_xcond = [w for w in my_list if w != x]
    return (meaningful_words_xcond)

test_data.loc[:,'meaningful_words_xcond'] = test_data.apply(remove_condition, axis=1)

def rejoin_words(row):
    my_list = row['meaningful_words_xcond']
    joined_words = ( " ".join(my_list))
    return (joined_words)

# def rejoin_words(condition_table):
#     my_list = condition_table['meaningful_words_xcond']
#     sent_str = ""
#     for i in my_list:
#         sent_str += str(i) + " "
#     joined_words = sent_str
#     return (joined_words)

test_data.loc[:,'joined_words'] = test_data.apply(rejoin_words, axis=1)

end_time=time.time()
print("總共執行了:  %d min  %d sec" % (int((end_time-start_time)/60),(end_time-start_time)%60))


# # Model
data = [train_data, test_data]
df = pd.concat(data)
print(df.shape)

import time
start_time=time.time()

import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

model = MultinomialNB()
kf=KFold(n_splits=20, shuffle=True)
tfidf_vectorizer = TfidfVectorizer()

predicted= []
expected = []
for train_index, test_index in kf.split(df.joined_words):
    x_train = np.array(df.joined_words)[train_index]
    y_train = np.array(df.Label)[train_index]
    x_test = np.array(df.joined_words)[test_index]
    y_test = np.array(df.Label)[test_index]
    
    model.fit(tfidf_vectorizer.fit_transform(df["joined_words"]),
                             df['Label'])
    expected.extend(y_test) 
    predicted.extend(model.predict(tfidf_vectorizer.transform(x_test)))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )

end_time=time.time()
print("總共執行了:  %d min  %d sec" % (int((end_time-start_time)/60),(end_time-start_time)%60))





