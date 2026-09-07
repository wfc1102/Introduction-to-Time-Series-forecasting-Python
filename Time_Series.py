# -*- coding: utf-8 -*-
"""
Time Series Analysis and Forecasting in Python
提取自: Time Series in Python.ipynb
共 104 个代码单元格
已修改: 注释掉 plt.show()，改为保存图像；添加中文字体设置
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图像输出目录
FIG_DIR = './figures'
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(fig_name):
    """保存当前图像到 figures 目录"""
    filepath = os.path.join(FIG_DIR, fig_name)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f'  [已保存] {filepath}')
    plt.close()


# ============================================================
# Cell 5 - 导入库
# ============================================================
import itertools
import numpy as np
plt.style.use('fivethirtyeight')

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from math import sqrt

import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import seaborn as sns

from random import random

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error


# ============================================================
# Cell 7 - 加载数据
# ============================================================
df = pd.read_csv('international-airline-passengers.csv', header=None)


# ============================================================
# Cell 8 - 设置列名
# ============================================================
df.columns = ['year', 'passengers']


# ============================================================
# Cell 9 - 查看前3行
# ============================================================
df.head(3)


# ============================================================
# Cell 10 - 数值统计
# ============================================================
df.describe()


# ============================================================
# Cell 11 - 非数值统计
# ============================================================
df.describe(include='O')


# ============================================================
# Cell 12 - 时间范围
# ============================================================
print('Time period start: {}\nTime period end: {}'.format(df.year.min(), df.year.max()))


# ============================================================
# Cell 13 - 列名
# ============================================================
df.columns


# ============================================================
# Cell 14 - 数据形状
# ============================================================
df.shape


# ============================================================
# Cell 17 - 转换日期格式
# ============================================================
df['year'] = pd.to_datetime(df['year'], format='%Y-%m')


# ============================================================
# Cell 19 - 设置时间索引
# ============================================================
y = df.set_index('year')


# ============================================================
# Cell 20 - 查看索引
# ============================================================
y.index


# ============================================================
# Cell 21 - 缺失值检查
# ============================================================
y.isnull().sum()


# ============================================================
# Cell 22 - 时序图
# ============================================================
y.plot(figsize=(15, 6))
save_fig('01_time_series_overview.png')
# plt.show()


# ============================================================
# Cell 24 - 直方图与KDE
# ============================================================
from pandas import Series
from matplotlib import pyplot
pyplot.figure(1)
pyplot.subplot(211)
y.passengers.hist()
pyplot.subplot(212)
y.passengers.plot(kind='kde')
save_fig('02_histogram_kde.png')
# pyplot.show()


# ============================================================
# Cell 26 - 按年箱线图
# ============================================================
fig, ax = plt.subplots(figsize=(15, 6))
sns.boxplot(x=y.passengers.index.year, y=y.passengers, ax=ax)
save_fig('03_yearly_boxplot.png')
# plt.show()


# ============================================================
# Cell 28 - 季节分解（乘法模型）
# ============================================================
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
fig = decomposition.plot()
save_fig('04_seasonal_decomposition.png')
# plt.show()


# ============================================================
# Cell 30 - 简单折线图
# ============================================================
plt.plot(y)
save_fig('05_simple_lineplot.png')
# plt.show()


# ============================================================
# Cell 33 - ACF 和 PACF 图
# ============================================================
pyplot.figure()
pyplot.subplot(211)
plot_acf(y.passengers, ax=pyplot.gca(), lags=30)
pyplot.subplot(212)
plot_pacf(y.passengers, ax=pyplot.gca(), lags=30)
save_fig('06_acf_pacf_original.png')
# pyplot.show()


# ============================================================
# Cell 35 - 滚动统计量
# ============================================================
# Determing rolling statistics
rolmean = y.rolling(window=12).mean()
rolstd = y.rolling(window=12).std()

# Plot rolling statistics:
orig = plt.plot(y, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
save_fig('07_rolling_statistics.png')
# plt.show(block=False)


# ============================================================
# Cell 37 - 导入 ADF 检验
# ============================================================
from statsmodels.tsa.stattools import adfuller


# ============================================================
# Cell 38 - Dickey-Fuller 检验
# ============================================================
# Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test:')
dftest = adfuller(y.passengers, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)


# ============================================================
# Cell 39 - 平稳性检验函数
# ============================================================
def test_stationarity(timeseries, fig_name='stationarity_test.png'):

    # Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    save_fig(fig_name)
    # plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


# ============================================================
# Cell 43 - 对数变换
# ============================================================
ts_log = np.log(y)
plt.plot(ts_log)
save_fig('08_log_transformed.png')
# plt.show()


# ============================================================
# Cell 47 - 移动平均
# ============================================================
moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
save_fig('09_moving_average.png')
# plt.show()


# ============================================================
# Cell 48 - 对数移动平均差分
# ============================================================
ts_log_moving_avg_diff = ts_log.passengers - moving_avg.passengers
ts_log_moving_avg_diff.head(12)


# ============================================================
# Cell 49 - 移动平均差分平稳性检验
# ============================================================
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff, fig_name='10_ma_diff_stationarity.png')


# ============================================================
# Cell 51 - 指数加权移动平均
# ============================================================
expwighted_avg = ts_log.ewm(halflife=12).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
save_fig('11_ewma.png')
# plt.show()


# ============================================================
# Cell 52 - 指数加权平均差分
# ============================================================
ts_log_ewma_diff = ts_log.passengers - expwighted_avg.passengers
test_stationarity(ts_log_ewma_diff, fig_name='12_ewma_diff_stationarity.png')


# ============================================================
# Cell 55 - 一阶差分
# ============================================================
ts_log_diff = ts_log.passengers - ts_log.passengers.shift()
plt.plot(ts_log_diff)
save_fig('13_first_difference.png')
# plt.show()


# ============================================================
# Cell 56 - 差分后平稳性检验
# ============================================================
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff, fig_name='14_diff_stationarity.png')


# ============================================================
# Cell 58 - 季节分解（加法模型）
# ============================================================
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
save_fig('15_additive_decomposition.png')
# plt.show()


# ============================================================
# Cell 59 - 残差平稳性检验
# ============================================================
ts_log_decompose = residual.squeeze()
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose, fig_name='16_residual_stationarity.png')


# ============================================================
# Cell 62 - 导入 AR 模型
# ============================================================
from statsmodels.tsa.ar_model import AutoReg
from random import random


# ============================================================
# Cell 63 - 拟合 AR 模型
# ============================================================
# fit model
model = AutoReg(ts_log_diff, lags=12)
model_fit = model.fit()


# ============================================================
# Cell 64 - AR 模型拟合效果
# ============================================================
plt.plot(ts_log_diff)
plt.plot(model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f' % np.nansum((model_fit.fittedvalues - ts_log_diff) ** 2))
save_fig('17_ar_fit.png')
# plt.show()


# ============================================================
# Cell 67 - AR 预测差分序列
# ============================================================
predictions_ARIMA_diff = pd.Series(model_fit.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


# ============================================================
# Cell 69 - 累积求和还原
# ============================================================
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# ============================================================
# Cell 71 - 还原到对数尺度
# ============================================================
predictions_ARIMA_log = pd.Series(ts_log.passengers.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# ============================================================
# Cell 73 - 指数还原到原始尺度
# ============================================================
predictions_ARIMA = np.exp(predictions_ARIMA_log)


# ============================================================
# Cell 74 - AR 模型最终效果
# ============================================================
plt.plot(y.passengers)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f' % np.sqrt(np.nansum((predictions_ARIMA - y.passengers) ** 2) / len(y.passengers)))
save_fig('18_ar_final.png')
# plt.show()


# ============================================================
# Cell 76 - 导入评估指标
# ============================================================
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error


# ============================================================
# Cell 78 - R² 分数
# ============================================================
r2_score(y.passengers, predictions_ARIMA)


# ============================================================
# Cell 80 - 平均绝对误差
# ============================================================
mean_absolute_error(y.passengers, predictions_ARIMA)


# ============================================================
# Cell 82 - 中位数绝对误差
# ============================================================
median_absolute_error(y.passengers, predictions_ARIMA)


# ============================================================
# Cell 84 - 均方误差
# ============================================================
mean_squared_error(y.passengers, predictions_ARIMA)


# ============================================================
# Cell 86 - 均方对数误差
# ============================================================
mean_squared_log_error(y.passengers, predictions_ARIMA)


# ============================================================
# Cell 88 - MAPE 函数
# ============================================================
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ============================================================
# Cell 89 - 计算 MAPE
# ============================================================
mean_absolute_percentage_error(y.passengers, predictions_ARIMA)


# ============================================================
# Cell 91 - 综合评估函数
# ============================================================
def evaluate_forecast(y, pred):
    results = pd.DataFrame({'r2_score': r2_score(y, pred)}, index=[0])
    results['mean_absolute_error'] = mean_absolute_error(y, pred)
    results['median_absolute_error'] = median_absolute_error(y, pred)
    results['mse'] = mean_squared_error(y, pred)
    results['msle'] = mean_squared_log_error(y, pred)
    results['mape'] = mean_absolute_percentage_error(y, pred)
    results['rmse'] = np.sqrt(results['mse'])
    return results


# ============================================================
# Cell 92 - 综合评估 AR 模型
# ============================================================
evaluate_forecast(y.passengers, predictions_ARIMA)


# ============================================================
# Cell 95 - MA(1) 模型
# ============================================================
# MA example (ARMA 已移除，改用 ARIMA(d=0))
from random import random

# fit model
model = ARIMA(ts_log_diff, order=(0, 0, 1))
model_fit = model.fit()


# ============================================================
# Cell 96 - MA 模型摘要
# ============================================================
model_fit.summary()


# ============================================================
# Cell 97 - MA 模型拟合效果
# ============================================================
plt.plot(ts_log_diff)
plt.plot(model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f' % np.nansum((model_fit.fittedvalues - ts_log_diff) ** 2))
save_fig('19_ma_fit.png')
# plt.show()


# ============================================================
# Cell 99 - ARMA(2,1) 模型
# ============================================================
# ARMA example (ARMA 已移除，改用 ARIMA(d=0))
from random import random

# fit model
model = ARIMA(ts_log_diff, order=(2, 0, 1))
model_fit = model.fit()


# ============================================================
# Cell 100 - ARMA 模型摘要
# ============================================================
model_fit.summary()


# ============================================================
# Cell 101 - ARMA 拟合效果
# ============================================================
plt.plot(ts_log_diff)
plt.plot(model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f' % np.nansum((model_fit.fittedvalues - ts_log_diff) ** 2))
save_fig('20_arma_fit.png')
# plt.show()


# ============================================================
# Cell 103 - 原始序列一阶差分
# ============================================================
ts = y.passengers - y.passengers.shift()
ts.dropna(inplace=True)


# ============================================================
# Cell 105 - 差分后 ACF/PACF
# ============================================================
pyplot.figure()
pyplot.subplot(211)
plot_acf(ts, ax=pyplot.gca(), lags=30)
pyplot.subplot(212)
plot_pacf(ts, ax=pyplot.gca(), lags=30)
save_fig('21_diff_acf_pacf.png')
# pyplot.show()


# ============================================================
# Cell 107 - 划分训练集和验证集
# ============================================================
# divide into train and validation set
train = y[:int(0.75 * (len(y)))]
valid = y[int(0.75 * (len(y))):]

# plotting the data
train['passengers'].plot()
valid['passengers'].plot()
save_fig('22_train_valid_split.png')
# plt.show()


# ============================================================
# Cell 108 - ARIMA(1,1,1) 模型
# ============================================================
# ARIMA example (新版 statespace ARIMA，顶部已导入)
from sklearn.metrics import mean_squared_error
from math import sqrt

# fit model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()


# ============================================================
# Cell 109 - ARIMA 模型摘要
# ============================================================
model_fit.summary()


# ============================================================
# Cell 110 - ARIMA 预测
# ============================================================
start_index = valid.index.min()
end_index = valid.index.max()

# Predictions
predictions = model_fit.predict(start=start_index, end=end_index)


# ============================================================
# Cell 111 - ARIMA 预测误差
# ============================================================
# report performance
mse = mean_squared_error(y[start_index:end_index], predictions)
rmse = sqrt(mse)
print('RMSE: {}, MSE:{}'.format(rmse, mse))


# ============================================================
# Cell 112 - ARIMA 预测可视化
# ============================================================
plt.plot(y.passengers)
plt.plot(predictions, color='red')
plt.title('RMSE: %.4f' % rmse)
save_fig('23_arima_forecast.png')
# plt.show()


# ============================================================
# Cell 114-118 - 预测还原（新版 statespace ARIMA 无需手动还原）
# ============================================================
# 注意：新版 statsmodels.tsa.arima.model.ARIMA 的 predict()
# 直接返回原始尺度的预测值，旧版的差分→cumsum→加回初始值的还原步骤不再需要
predictions_ARIMA_log = pd.Series(predictions, index=valid.index)
print(predictions_ARIMA_log.head())


# ============================================================
# Cell 120 - 还原后可视化
# ============================================================
plt.plot(y.passengers)
plt.plot(predictions_ARIMA_log)
plt.title('RMSE: %.4f' % np.sqrt(np.nansum((predictions_ARIMA_log - ts) ** 2) / len(ts)))
save_fig('24_arima_restored.png')
# plt.show()


# ============================================================
# Cell 121 - 评估还原后预测
# ============================================================
evaluate_forecast(y[start_index:end_index], predictions_ARIMA_log)


# ============================================================
# Cell 123 - auto_arima 自动选参
# ============================================================
# building the model (auto_arima 顶部已从 pmdarima 导入)
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)


# ============================================================
# Cell 124 - auto_arima 预测
# ============================================================
forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

# plot the predictions for validation set
plt.plot(y.passengers, label='Train')
# plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend()
save_fig('25_auto_arima.png')
# plt.show()


# ============================================================
# Cell 125 - auto_arima 评估
# ============================================================
evaluate_forecast(valid, forecast)


# ============================================================
# Cell 127 - SARIMA 模型
# ============================================================
# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX

# fit model
model = SARIMAX(train, order=(3, 1, 3), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()


# ============================================================
# Cell 128 - SARIMA 预测
# ============================================================
start_index = valid.index.min()
end_index = valid.index.max()

# Predictions
predictions = model_fit.predict(start=start_index, end=end_index)


# ============================================================
# Cell 129 - SARIMA 误差
# ============================================================
# report performance
mse = mean_squared_error(y[start_index:end_index], predictions)
rmse = sqrt(mse)
print('RMSE: {}, MSE:{}'.format(rmse, mse))


# ============================================================
# Cell 130 - SARIMA 可视化
# ============================================================
plt.plot(y)
plt.plot(predictions)
plt.title('RMSE: %.4f' % rmse)
save_fig('26_sarima.png')
# plt.show()


# ============================================================
# Cell 131 - SARIMA 评估
# ============================================================
evaluate_forecast(y[start_index:end_index], predictions)


# ============================================================
# Cell 133 - auto_arima 季节性模式
# ============================================================
# building the model (auto_arima 顶部已从 pmdarima 导入)
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True,
                    seasonal=True, m=6, stepwise=True)
model.fit(train)


# ============================================================
# Cell 134 - 预测索引设置
# ============================================================
start_index = valid.index.min()
end_index = valid.index.max()

# Predictions
pred = model.predict()


# ============================================================
# Cell 135 - 季节性 auto_arima 预测
# ============================================================
pred = model.predict(n_periods=len(valid))
pred = pd.DataFrame(pred, index=valid.index, columns=['Prediction'])


# ============================================================
# Cell 136 - 预测并可视化
# ============================================================
forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

# plot the predictions for validation set
plt.plot(y.passengers, label='Train')
# plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend()
save_fig('27_seasonal_auto_arima.png')
# plt.show()


# ============================================================
# Cell 137 - 季节性 auto_arima 评估
# ============================================================
evaluate_forecast(y[start_index:end_index], forecast)


# ============================================================
# Cell 139 - 网格搜索参数组合
# ============================================================
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 6) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# ============================================================
# Cell 140 - 网格搜索最优 SARIMA（AIC 准则）
# ============================================================
min_aic = 999999999
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train,
                                              order=param,
                                              seasonal_order=param_seasonal,
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)

            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

            # Check for best model with lowest AIC
            if results.aic < min_aic:
                min_aic = results.aic
                min_aic_model = results
        except:
            continue


# ============================================================
# Cell 141 - 最优模型摘要
# ============================================================
min_aic_model.summary()


# ============================================================
# Cell 142 - 最优模型预测（含置信区间）
# ============================================================
start_index = valid.index.min()
end_index = valid.index.max()

# Predictions
pred = min_aic_model.get_prediction(start=start_index, end=end_index, dynamic=False)


# ============================================================
# Cell 143 - 预测与置信区间可视化
# ============================================================
pred_ci = pred.conf_int()
ax = y['1949':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers')
plt.legend()
save_fig('28_sarima_grid_forecast.png')
# plt.show()


# ============================================================
# Cell 145 - 模型诊断图
# ============================================================
results.plot_diagnostics(figsize=(16, 8))
save_fig('29_sarima_diagnostics.png')
# plt.show()


# ============================================================
# Cell 146 - 计算 MSE
# ============================================================
y_forecasted = pred.predicted_mean.values
y_truth = y[start_index:end_index].passengers.values
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# ============================================================
# Cell 147 - 计算 RMSE
# ============================================================
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# ============================================================
# Cell 148 - 最优模型综合评估
# ============================================================
evaluate_forecast(y_truth, y_forecasted)


# ============================================================
# Cell 153 - 训练集预览
# ============================================================
train.head()


# ============================================================
# Cell 154 - Prophet 格式转换
# ============================================================
train_prophet = pd.DataFrame()
train_prophet['ds'] = train.index
train_prophet['y'] = train.passengers.values


# ============================================================
# Cell 155 - Prophet 数据预览
# ============================================================
train_prophet.head()


# ============================================================
# Cell 156 - 拟合 Prophet 模型
# ============================================================
from prophet import Prophet

# instantiate Prophet with only yearly seasonality as our data is monthly
model = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
model.fit(train_prophet)  # fit the model with your dataframe


# ============================================================
# Cell 157 - 构建未来日期框架
# ============================================================
# predict for five months in the future and MS - month start is the frequency
future = model.make_future_dataframe(periods=36, freq='MS')
future.tail()


# ============================================================
# Cell 158 - 预测结果列名
# ============================================================
forecast.columns


# ============================================================
# Cell 159 - Prophet 预测
# ============================================================
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# ============================================================
# Cell 160 - Prophet 预测可视化
# ============================================================
fig = model.plot(forecast)
# plot the predictions for validation set
plt.plot(valid, label='Valid', color='red', linewidth=2)
plt.legend()
save_fig('30_prophet_forecast.png')
# plt.show()


# ============================================================
# Cell 161 - Prophet 成分分解图
# ============================================================
model.plot_components(forecast)
save_fig('31_prophet_components.png')
# plt.show()


# ============================================================
# Cell 162 - 准备 Prophet 评估数据
# ============================================================
y_prophet = pd.DataFrame()
y_prophet['ds'] = y.index
y_prophet['y'] = y.passengers.values


# ============================================================
# Cell 163 - 设置索引对齐
# ============================================================
y_prophet = y_prophet.set_index('ds')
forecast_prophet = forecast.set_index('ds')


# ============================================================
# Cell 164 - Prophet 模型评估
# ============================================================
evaluate_forecast(y_prophet.y[start_index:end_index], forecast_prophet.yhat[start_index:end_index])

print('\n' + '='*60)
print('全部图像已保存至目录: ./figures/')
print('='*60)
