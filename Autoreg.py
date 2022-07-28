from pandas import read_csv
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import re

raw_data = pd.read_csv('./arima/data_3240_20220722.csv',encoding='cp949', header=0)
# split dataset

#################### 날짜 지정 및 날짜 순서대로 정렬 #########
raw_data['일자']=pd.to_datetime(raw_data['일자'])
print(raw_data)
raw_data= raw_data.sort_values(by='일자', ascending=True)
raw_data.set_index('일자', inplace=True)
raw_data.info()


################### plot 보기 #########################
raw_data = raw_data['종가']
raw_data.plot()
pyplot.show()


minmaxscaler = MinMaxScaler()

print(raw_data.shape)
raw_data =raw_data.values.reshape(-1,1)
scaled_data = minmaxscaler.fit_transform(raw_data)
print('scaled_data출력:',scaled_data)
################## train 셋, test 셋 분배 ##################
train, test = scaled_data[1:len(scaled_data)-100], scaled_data[len(scaled_data)-100:]
print('train:', train)
print('test:', test)
# train autoregression
model = AutoReg(train, 500)
model_fit = model.fit()
# print('Lag: %s' % model_fit.)
print('Coefficients: %s' % model_fit.params) # 각각의 계수
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
# predictions.reset_index(inplace=True, drop=True)
# data_monthly = train.resample('M').sum()
# print('index_predictions:', predictions.index)
for i in range(len(predictions)): # 테스트값과 예측 값의 비교
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
# pyplot.xticks(data_monthly, data_monthly)
pyplot.plot(predictions, color='red')
pyplot.show()

minmaxscaler_close = MinMaxScaler()
raw_data.reshape(-1)


minmaxscaler_close.fit_transform(raw_data)
predictions = np.reshape(predictions,(-1,1))

print(predictions.shape)
tomorrow_predicted_value = minmaxscaler_close.inverse_transform(predictions) # inverse_transform은 스케일링 했던것을 다시 복원시키는 것
print('%d원'%tomorrow_predicted_value[0][0])