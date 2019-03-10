# -*- coding: utf-8 -*-
# !/usr/bin/env bash

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import os
import sys

os.chdir('E:/sofasofa/public_bike_usage')
print(os.getcwd())

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submit = pd.read_csv('data/sample_submit.csv')

train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

y_train = train.pop('y')

gbdt = GradientBoostingRegressor()

# gbdt.fit(train, y_train)
# y_pred = gbdt.predict(test)
#
# submit['y'] = y_pred
# print(submit.count())
# submit.to_csv('my_LR_prediction.csv', index=False)