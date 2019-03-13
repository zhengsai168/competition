# -*- coding: utf-8 -*-
# !/usr/bin/env bash

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import os
import sys
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

os.chdir('E:/sofasofa/public_bike_usage')
print(os.getcwd())

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submit = pd.read_csv('data/sample_submit.csv')

train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

# 特征相加减
train.eval('x=temp_1-temp_2',inplace=True)
test.eval('x=temp_1-temp_2',inplace=True)

# weather 做one-hot 编码
train = pd.get_dummies(train, columns= ['weather'])
test = pd.get_dummies(test, columns=['weather'])

# print(train.head())
y_train = train.pop('y')

param_grid = {
    #'subsample': np.linspace(0.3, 1, 5),
    #'colsample_bytree': np.arange(0.3,1,5)
    #'min_child_weight':np.arange(1,16)
    #'reg_alpha':[i*0.0 for i in range(10)]
    #'gamma':np.linspace(0,0.2,30)
    #'learning_rate':np.linspace(0.05,0.2,10)
    #'n_estimators': [i*100 for i in np.arange(1,11)]
    'subsample': np.linspace(0.8,1,20)
    #'colsample_bytree': [i*0.05 for i in np.arange(16,21)]
}

random_param_dict = {
    # 'max_depth': [3,4,5,6],
    # 'learning_rate': np.linspace(0.02,0.2,10),
    # 'n_estimators': range(100,1001,100)
    # 'gamma': np.linspace(0.03,0.06,60),
    # 'min_child_weight': np.arange(5,15),
    # 'max_delta_step': np.arange(0,10),
    # 'sub_sample': np.linspace(0.7,1,10),
    # 'colsample_bytree': np.linspace(0.7,1,10),
    # 'colsample_bylevel': np.linspace(0.7,0.75,30),
    # 'reg_alpha': np.linspace(0.9,1.1,30),
    # 'reg_lambda': np.linspace(0.1,1,10),
    # 'scale_pos_weight': np.linspace(0.1,10,10)
        # max_depth=3, learning_rate=0.1, n_estimators=100,
        #          silent=True, objective="reg:linear",
        #          nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
        #          subsample=1, colsample_bytree=1, colsample_bylevel=1,
        #          reg_alpha=0, reg_lambda=1, scale_pos_weight=1
}



# xgbr = XGBRegressor(max_depth = 5,learning_rate=0.09,n_estimators=200,
#                     reg_alpha=0.05,gamma=0.041379310344827593)
                    #,subsample=0.8,colsample_bytree=0.8)
                    #,min_child_weight=10,
                    #subsample=0.7,colsample_bytree=0.3,reg_alpha=0.001,gamma=0.07)
# xgbr = XGBRegressor(max_depth=5,n_estimators=200,learning_rate=0.086610169491525418,
#                     reg_alpha=0.96896551724137936,gamma=0.03,
#                     subsample=0.85, colsample_bylevel=0.71551724137931028,)
rf_param_dict = {
    # 'n_estimators': range(100,1001,100),
    #'max_depth': range(3,15),
    # 'min_samples_split': range(10,101,10),
    # 'min_samples_leaf': range(1,11),
    # 'max_features': range(2,8)
}

rf = RandomForestRegressor(n_estimators=800, max_depth=12,min_samples_split=10,min_samples_leaf=2,
                           max_features=6, )
xgbr = XGBRegressor(n_estimators=900,learning_rate=0.02,max_depth=5,min_child_weight=15,gamma=0.2,subsample=0.75)
grid = GridSearchCV(xgbr,random_param_dict,scoring='neg_mean_squared_error')
# grid = RandomizedSearchCV(xgbr,random_param_dict,n_iter=10,scoring='neg_mean_squared_error')
# x = []
# y = []
# for i in range(200,401,10):
#     xgbr = XGBRegressor(n_estimators=i,learning_rate=0.05,max_depth=5,min_child_weight=7,gamma=0.2,subsample=0.8)
#     grid = GridSearchCV(xgbr,rf_param_dict,scoring='neg_mean_squared_error')
#     grid.fit(train, y_train)
#     x.append(i)
#     y.append(grid.best_score_)
    #print(grid.best_params_)
    #print(grid.best_score_)



# plt.plot(x,y)
# plt.show()

grid.fit(train, y_train)
rf.fit(train, y_train)
print(grid.best_params_)
print(grid.best_score_)
y_pred_1 = grid.predict(test)
y_pred_1 = list(map(lambda x: x if x >= 0 else 0, y_pred_1))

y_pred_2 = rf.predict(test)
y_pred_2 = list(map(lambda x: x if x >= 0 else 0, y_pred_2))

y_pred = [(t1+t2)/2.0 for t1,t2 in zip(y_pred_1,y_pred_2)]

submit['y'] = y_pred_1
submit.to_csv('my_xgb_prediction_5.csv', index=False)