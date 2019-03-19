# -*- coding: utf-8 -*-
# !/usr/bin/env bash

import numpy as np
# x=np.array([[0],[1]])
# y=np.array([0,1])
# from sklearn.naive_bayes import MultinomialNB,BernoulliNB
# clf1=MultinomialNB()
# clf1.fit(x,y)
# clf2=BernoulliNB()
# clf2.fit(x,y)
# t=np.array([[1]])
# print(np.exp(clf1.class_log_prior_))
# print(np.exp(clf2.class_log_prior_))
#
# print(np.exp(clf1.feature_log_prob_))
# print(np.exp(clf2.feature_log_prob_))
#
# print(clf1.class_count_)
# print(clf2.class_count_)
#
# print(clf1.feature_count_)
# print(clf2.feature_count_)
#
# print(clf1.predict_proba(t))
# print(clf2.predict_proba(t))

# import time
#
# time_start=time.time()
# s = 0
# for i in range(int(1e8)):
#     s=s+i
# time_end=time.time()
# print('totally cost',time_end-time_start)
import os
print ('/'.join(os.getcwd().split('\\')))
def go(s):
    return ('/'.join(s.split('\\')))

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

p = '/home/ai_platform/project'

li = []

listdir(p,li)
for n in li:
    print(go(n))