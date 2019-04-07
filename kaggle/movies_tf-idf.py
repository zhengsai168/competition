# coding: utf-8
import pandas as pd
import os
from lxml import etree
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier


#load the data
path = "E:\kaggle\movies"
t_set_df = pd.read_csv(os.path.join(path,"labeledTrainData.tsv"), header=0, sep='\t')
test_df = pd.read_csv(os.path.join(path,"testData.tsv"), header=0, sep='\t')
t_set_pre = t_set_df['review']
test_pre = test_df['review']
t_set = []
test = []
t_label = t_set_df['sentiment']

#data preprocessing(remove the html labels)
def review2wordlist(review):
    html = etree.HTML(review, etree.HTMLParser())
    review = html.xpath('string(.)').strip()
    review = re.sub("[^a-zA-Z]", " ", review)
    wordlist = review.lower().split()
    return wordlist
for i in range(len(t_set_pre)):
    words = review2wordlist(t_set_pre[i])
    t_set.append(" ".join(words))
for i in range(len(test_pre)):
    words = review2wordlist(test_pre[i])
    test.append(" ".join(words))


#vectorize sentences with words' TF-IDF value
all_x = t_set+test
tfv = TFIV(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
tfv.fit(all_x)
all_x = tfv.transform(all_x)
train_len = len(t_set)
x_train = all_x[:train_len] #<25000x309819 sparse matrix of type '<class 'numpy.float64'>'with 3429925 stored elements in Compressed Sparse Row format>
x_test = all_x[train_len:]


#model_1: logistic regression
y_train = t_set_df['sentiment']
lr = LogisticRegression(C=30)
grid_value = {'solver':['sag','liblinear','lbfgs']}
model_lr = GridSearchCV(lr, cv=20, scoring='roc_auc', param_grid=grid_value)
model_lr.fit(x_train, y_train)
print(model_lr.cv_results_)  #the best score is 0.96462 with sag


#model_2: naive bayes
model_nb = MultinomialNB()
model_nb.fit(x_train, y_train)
print("naive bayes score: ", np.mean(cross_val_score(model_nb, x_train, y_train, cv=20, scoring='roc_auc')))  #0.94963712


#model_3: SGDClassifier (SVM with linear knernel)
model_sgd = SGDClassifier(loss='modified_huber')
model_sgd.fit(x_train, y_train)
print("SGD score: ", np.mean(cross_val_score(model_sgd, x_train, y_train, cv=20, scoring='roc_auc'))) #0.964716288


# write the result to csv
lr_result = model_lr.predict(x_test)
lr_df = pd.DataFrame({'id':test_df['id'], 'sentiment':lr_result})
lr_df.to_csv(os.path.join(path,"LR_result.csv"), index=False)

nb_result = model_nb.predict(x_test)
nb_df = pd.DataFrame({'id':test_df['id'],'sentiment':nb_result})
nb_df.to_csv(os.path.join(path,"NB_result.csv"), index=False)

sgd_result = model_nb.predict(x_test)
sgd_df = pd.DataFrame({'id':test_df['id'],'sentiment':sgd_result})
sgd_df.to_csv(os.path.join(path,"SGD_result.csv"), index=False)
