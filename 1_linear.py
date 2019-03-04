import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import sys

os.chdir('E:\\sofasofa\\public_bike_usage')
os.getcwd()

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submit = pd.read_csv('data/sample_submit.csv')

train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

y_train = train.pop('y')

reg = LinearRegression()
reg.fit(train, y_train)
y_pred = reg.predict(test)

y_pred = list(map(lambda x: x if x >= 0 else 0, y_pred))
# 输出预测结果至my_LR_prediction.csv
submit['y'] = y_pred
submit.to_csv('my_LR_prediction.csv', index=False)