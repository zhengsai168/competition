# -*- coding: utf-8 -*-
# !/usr/bin/env bash

import numpy as np
x=np.array([[0],[1]])
y=np.array([0,1])
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
clf1=MultinomialNB()
clf1.fit(x,y)
clf2=BernoulliNB()
clf2.fit(x,y)
t=np.array([[1]])
print(np.exp(clf1.class_log_prior_))
print(np.exp(clf2.class_log_prior_))

print(np.exp(clf1.feature_log_prob_))
print(np.exp(clf2.feature_log_prob_))

print(clf1.class_count_)
print(clf2.class_count_)

print(clf1.feature_count_)
print(clf2.feature_count_)

print(clf1.predict_proba(t))
print(clf2.predict_proba(t))

