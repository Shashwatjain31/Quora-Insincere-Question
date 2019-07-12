import os
import string
import random
import operator
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

train_df= pd.read_csv("256/train.csv")


# UNDERSAMPLING

sincerequestions=train_df[:][train_df['target']==0]
insincerequestions=train_df[:][train_df['target']==1]
sincerequestions_under=sincerequestions.sample(len(insincerequestions))
train_under = pd.concat([sincerequestions_under,insincerequestions], axis=0)
train_temp = train_under.drop(['target'],axis=1)


# LOADING EMBEDDINGS FROM GLOVE

embeddings_index = {}
f = open('256/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# PREPROCESSING

punctuations = string.punctuation
stopwords = list(STOP_WORDS)

parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

train_temp["question_text"] = train_temp["question_text"].apply(lambda x: spacy_tokenizer(x))


# SPLITTING TRAIN AND TEST

train_x, val_x,train_y,val_y = train_test_split(train_temp,train_under['target'],test_size=.20, random_state=0)

# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)

X_train = [text_to_array(X_text) for X_text in tqdm(train_x["question_text"])]
X_val = [text_to_array(X_text) for X_text in tqdm(val_x["question_text"])]

# RESHAPING 3D ARRAY

import numpy as np
trainvects=np.asarray(X_train)
valvects=np.asarray(X_val)
nsamples, nx, ny = trainvects.shape
X_train = trainvects.reshape((nsamples,nx*ny))
msamples, mx, my = valvects.shape
X_val = valvects.reshape((msamples,mx*my))


# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train,train_y)
y_pred1=logr.predict(X_val)


# XG BOOST CLASSIFIER

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,train_y)
y_pred2 = xgb.predict(X_val)


# BERNOULLI NAIVE BAYES CLASSIFIER

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB(alpha=0.01)
bnb.fit(X_train,train_y)
y_pred3 = bnb.predict(X_val)


# EVALUATION USING ACCURACY SCORE and F1 SCORE
from sklearn.metrics import f1_score, balanced_accuracy_score

def evaluate(model,y_predict):
    print(model)
    f1=f1_score(val_y,y_predict)
    accuracy= balanced_accuracy_score(val_y,y_predict)
    print("F1 score:",f1)
    print("Accuracy:",accuracy)
    return f1,accuracy

f1_logr,acc_logr=evaluate("STANDARD VECTOR CLASSIFIER",y_pred1)

f1_xgb,acc_xgb=evaluate("XGBOOST CLASSIFIER",y_pred2)

f1_bnb,acc_bnb=evaluate("BERNOULLI NAIVE BAYES",y_pred3)


# EVALUATION GRAPH
import matplotlib.pyplot as plt

objects = ('STANDARD VECTOR CLASSIFIER', 'XGBOOST CLASSIFIER', 'BERNOULLI NAIVE BAYES')
y_pos = np.arange(len(objects))
performance1 = [f1_logr,f1_xgb,f1_bnb]
performance2=[acc_logr,acc_xgb,acc_bnb]

plt.bar(y_pos, performance1, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('F1 score')
plt.title('Classifier')

plt.show()

plt.bar(y_pos, performance2, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Classifier')

plt.show()
