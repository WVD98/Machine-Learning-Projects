
from pathlib import Path
from warnings import simplefilter

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

simplefilter("ignore")


# Creat data frame 

data = pd.read_csv(r'C:\Users\mathe\Desktop\Machine_Learning_Projects\Machine-Learning-Projects\air_traffic.csv',sep=';',
  parse_dates={'Date': ['Year','Month']},
    index_col='Date',
    )

y = data.loc[:, 'Pax'].dropna().to_period('M')

dataFrame = pd.DataFrame({
    'y':y,
    'K_1':y.shift(1),
    'K_2':y.shift(2),
    'K_3':y.shift(3),
    'K_4':y.shift(4),
    'K_5':y.shift(5),
    'K_6':y.shift(6),
    'K_7':y.shift(7),
    'K_8':y.shift(8),
    'K_9':y.shift(9),
    'K_10':y.shift(10),
    'K_11':y.shift(11),
    'K_12':y.shift(12),
    'K_13':y.shift(13),
    'K_14':y.shift(14),
    'K_15':y.shift(15),
    'K_16':y.shift(16),
    'K_17':y.shift(17),
    'K_18':y.shift(18),
    'K_19':y.shift(19),
    'K_20':y.shift(20),
    'K_21':y.shift(21),
    'K_22':y.shift(22),
    'K_23':y.shift(23),
    'K_24':y.shift(24),

})



X = dataFrame.loc[:, ['K_1','K_2','K_3','K_4','K_5','K_6','K_7','K_8','K_9']]
# X = dataFrame.loc[:, ['K_1']]
X.dropna(inplace=True)  # drop missing values in the feature set
y = dataFrame.loc[:, 'y']  # create the target
y, X = y.align(X, join='inner')  # drop corresponding values in target


train_X = X[:204]
train_y = y[:204]
test_X = X[204:]
test_y = y[204:]

model = LinearRegression()
model.fit(train_X, train_y)

y_pred = pd.Series(model.predict(train_X), index=train_y.index)
y_fore = pd.Series(model.predict(test_X), index=test_y.index)

# y_pred = pd.Series(model.predict(X_train), index=y_train.index)



fig, ax = plt.subplots()


ax = train_y.plot(color = 'r', label='Train')
ax = test_y.plot(color = 'b', label='Test')
ax = y_pred.plot(ax=ax, color = 'g', label='Data')
_ = y_fore.plot(ax=ax, color='black', label='Forecast')

#Erro RSME

def rmse(y, y_predict):  # calcula raiz quadrada do erro quadrático médio
    e = y - y_predict # erro entre estimado e real
    rmse = np.sqrt(sum(e**2)/len(e)) # raiz quadrada do erro quadrático médio
    return rmse

print(rmse(y_pred,y_fore))
print(y_pred)
print(y_fore)
print(1)

# ax = y.plot(color = 'y')
# ax = y_pred.plot()

ax.set_ylabel('Voos')
ax.set_xlabel('K')
ax.set_title('Número de voos Nacionais + Internacionais')
plt.legend()
plt.show()

