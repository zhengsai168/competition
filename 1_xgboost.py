# -*- coding: utf-8 -*-
# !/usr/bin/env bash

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import os
import sys
from sklearn.model_selection import GridSearchCV

os.chdir('E:/sofasofa/public_bike_usage')
print(os.getcwd())

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submit = pd.read_csv('data/sample_submit.csv')

train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

y_train = train.pop('y')

param_grid = dict(
    max_depth = [3,4, 5, 6, 7,8,9,10],
    learning_rate = np.linspace(0.03, 0.3, 10),
    n_estimators = [100, 200]
)

xgbr = XGBRegressor()
grid = GridSearchCV(xgbr,param_grid,cv=10,scoring='neg_mean_squared_error')

grid.fit(train, y_train)
print(grid.best_params_)
y_pred = grid.predict(test)

submit['y'] = y_pred
submit.to_csv('my_xgb_prediction.csv', index=False)