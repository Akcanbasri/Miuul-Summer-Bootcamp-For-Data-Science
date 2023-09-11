import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# import data
advertising = pd.read_csv("./datasets/advertising.csv")
df = advertising.copy()

df.head()
df.shape
# simple linear regression

X = df[["TV"]]
y = df[["sales"]]

# model = LinearRegression()

reg_model = LinearRegression().fit(X, y)
# y_hat = b + w * X

# bayes sabiti
reg_model.intercept_[0]

#  tv değişkeni agirlik degeri
reg_model.coef_[0][0]


# Tahömin etme işlemi
# 150 birim TV harcaması olduğunda satışların tahmini değeri nedir?
tahmin = reg_model.intercept_[0] + reg_model.coef_[0][0] * 500
print(tahmin)

df.describe().T

# modelin görselleştirilmesi

g = sns.regplot(x=X, y=y, ci=None, scatter_kws={"color": "r", "s": 9})
g.set_title("Model Denklemi: Sales = 7.03 + TV * 0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0);
