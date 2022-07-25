from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

series = read_csv('./arima/data_3240_20220722.csv',encoding='cp949', header=0, index_col=0)
# split dataset
X = series['종가']
print('출력:',X)
train, test = X[1:len(X)-300], X[len(X)-300:]
print(train)
# train autoregression
model = AutoReg(train, 500)
model_fit = model.fit()
# print('Lag: %s' % model_fit.)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
# for i in range(len(predictions)):
#     print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()