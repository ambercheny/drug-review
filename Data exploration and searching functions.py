#!/usr/bin/env python
# coding: utf-8

# The structure of the data is that a patient with a unique ID purchases a drug that meets his condition and writes a review and rating for the drug he/she purchased on the date. Afterwards, if the others read that review and find it helpful, they will click usefulCount, which will add 1 for the variable.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
import nltk
import re
from collections import Counter


# # Merge two datasets
train = pd.read_csv('drugsComTrain_raw.csv', index_col=0, parse_dates=True) #index_col=0 - remove index column, parse_date - to normalize date type
test = pd.read_csv('drugsComTest_raw.csv', index_col=0, parse_dates=True)

# inspect training and testing set 
print("shape of training set:", train.shape)
print("columns of training set:", train.columns, "\n")
print("shape of testing set:", test.shape)
print("columns of testing set:", test.columns, "\n") 
# make sure testing and training sets are with same columns
print (train.columns==test.columns)

# Concatenate two dataframes
data = [train, test]
df = pd.concat(data)
print(df.shape)

df.info()
df.isnull().sum()

#drop rows wih missing values
df = df.dropna()
df.isnull().sum() 


df.condition.unique()
df[df['condition']=='3</span> users found this comment helpful.'].head(3)

df = df[~df.condition.str.contains("</span>")]
print(df.shape)

df = df[~df.condition.str.contains("Not Listed / Othe")]
print(df.shape)
print("How many kinds of conditions are in the dataset:", df.condition.unique().size)

conditions = df.condition.value_counts().sort_values(ascending=False)
print("What are the top 10 discussed conditions:\n", conditions[:10])

drug_no = df.drugName.value_counts().sort_values(ascending=False)
print("How many kinds of drugs are in the dataset: %s \n" % (len(drug_no)))
print("What are the top 10 discussed drugs:\n", drug_no[:10])

drug = df.drugName.unique()
print(drug)


# # Drugs/Conditions

#1. Common drugs
drug_dn = df.drugName.value_counts() 
# returns a seires of conditions and its number
# convert series to dataframe
drug_dn = drug_dn.to_frame()
drug_dn = drug_dn.rename(columns = {'drugName':'drug_no'})

drug_dn = drug_dn.sort_values(by='drug_no', ascending=False)

drug_dn[:20]

plt.rcParams['figure.figsize'] = [12, 8]
drug_dn[:20].plot(kind='bar', color='red')
plt.title('Top 20 Most Common Drugs', fontsize = 20)
plt.xlabel('Drugs')
plt.ylabel('Count')
plt.savefig("drug20.png") # save as png


#2. Common conditions

condition_dn = df.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False) 
# # returns a seires of conditions and its number
# # convert series to dataframe
condition_dn = condition_dn.to_frame()
condition_dn = condition_dn.rename(columns = {'condition':'drug_no'})
condition_dn = condition_dn.rename(columns={'drugName':'drug_no'})

condition_dn = condition_dn.sort_values(by='drug_no', ascending=False)

condition_dn[:10]

plt.rcParams['figure.figsize'] = [12, 8]
condition_dn[:10].plot(kind='bar', color='blue')
plt.title('Top 10 Most Common Conditions', fontsize = 20)
plt.xlabel('Condition')
plt.ylabel('Count')
plt.savefig("condition10.png") # save as png

# # Functionns

#1. to enable users to find condition and drugs, we first make all words starts with a caplital letter
df['condition_low'] = df.condition.str.lower()
df['drug_low'] = df.drugName.str.lower()

def find_condition(query_with_lowercase):
    df1 = df.loc[df['condition_low'].str.startswith(query_with_lowercase), 'condition_low'].to_frame() # series to dataframe
    customized_condition = list(df1['condition_low'])
    return print(*customized_condition, sep = "\n")

def find_drug(query_with_lowercase):
    df1 = df.loc[df['drug_low'].str.startswith(query_with_lowercase), 'drug_low'].to_frame()
    customized_condition = list(df1['drug_low'])
    return print(*customized_condition, sep = "\n")

find_condition('r')
find_drug('f')


#2. How many and top n drugs for each condition
def drugs_for_condition (condition_query, number_of_drugs):
    mask = df['condition']==condition_query
    df1 = df[mask]

    x = df1['drugName'].value_counts()
    x = x.to_frame()
    answer1 = len(x)
    y = x.iloc[:number_of_drugs]
    u = list(y.index)
    joined_words = ( ", ".join(u))
    print("%s drugs are used for condition '%s'. \nTop %d discussed drugs for this condition are %s" %(answer1, condition_query, number_of_drugs, joined_words))

    drug=[]
    rate=[]
    for i in y.index:
        z = df.loc[df['drugName']==i]
        drug.append(z)
        w = z.rating.median()
        rate.append(w)
        list_labels = ['drug', 'rate']
        list_cols = [drug, rate]

        drug_rate = list(zip(list_labels, list_cols)) #把兩個list變成dictionary (ID as key, array_of_img as value) 
        drug_rate = dict(drug_rate)
        drug_rate_df = pd.DataFrame(drug_rate)
    

    drug_rate_df.plot(kind='bar', color='blue')
    plt.title('Top %d discussed drugs with rating (median)' %(number_of_drugs) , fontsize = 20)
    plt.xlabel('Drug')
    plt.ylabel('Rating (Median)')
    
drugs_for_condition('Pain',5)


#3. How many and top n conditions for each drug
def conditions_for_drug (drug_query, number_of_conditions):
    mask = df['drugName']==drug_query
    df1 = df[mask]

    x = df1['condition'].value_counts()
    x = x.to_frame()
    answer1 = len(x)
    y = x.iloc[:number_of_conditions]
    u = list(y.index)
    joined_words = ( ", ".join(u))
    print("%s conditions are used for drug '%s'. \nTop %d discussed conditions for this drug are %s" %(answer1, drug_query, number_of_conditions, joined_words))

    condition=[]
    rate=[]
    for i in y.index:
        z = df.loc[df['condition']==i]
        condition.append(z)
        w = z.rating.median()
        rate.append(w)
        list_labels = ['condition', 'rate']
        list_cols = [condition, rate]

        condition_rate = list(zip(list_labels, list_cols)) #把兩個list變成dictionary (ID as key, array_of_img as value) 
        condition_rate = dict(condition_rate)
        condition_rate_df = pd.DataFrame(condition_rate)
    

    condition_rate_df.plot(kind='bar', color='blue')
    plt.title('Top %d discussed conditions with rating (median)' %(number_of_conditions) , fontsize = 20)
    plt.xlabel('Drug')
    plt.ylabel('Rating (Median)')

conditions_for_drug('Tramadol', 5)


#4. Useful/Not useful comments of drug at rate n
def drug_rate_and_usefulness_for_condition(condition_query, rating_no, useful_comment_no, unuseful_comment_no):
    condition_table = df.loc[df['condition']==condition_query]
    condition_table_rate = condition_table.loc[condition_table['rating']==rating_no]
    condition_table_rate = condition_table_rate.sort_values (by=['usefulCount'], ascending = False)
    
    good = list (condition_table_rate['drugName'][:5])
    review_good = condition_table_rate.iloc[:5,2]
    review_good = review_good.to_frame()
    x = review_good.review.tolist()
    
    bad = list (condition_table_rate['drugName'][-5:])
    review_bad = condition_table_rate.iloc[:5,2]
    review_bad = review_bad.to_frame()
    y = review_bad.review.tolist()

    
    print ("Useful comments with drugs of rating %s: \n" %rating_no, good) 
    print("Useful comments: \n", x[:useful_comment_no], "\n")
    print ("Not useful comments with drugs of rating %s: \n" %rating_no, bad)
    print("Not useful comments: \n", y[:unuseful_comment_no])

drug_rate_and_usefulness_for_condition('Pain', 5, 1, 1)