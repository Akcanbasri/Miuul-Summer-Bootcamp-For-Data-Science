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
plt.ylim(bottom=0)


# Tahmin basarisi degerlendirme
# MSE 10.512652915656757
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)  # istediğimiz onun daha aşağıda olması

y.mean()
y.std()

# RMSE 3.2423221486546887
np.sqrt(mean_squared_error(y, y_pred))

# MAE 2.549806038927486
mean_absolute_error(y, y_pred)

# R-kare 0.611875050850071 ini açıklayabiliyor
reg_model.score(X, y)  # bagımsız değişkenlerin bağımlı değişkenleri açıklama yüzdesi


###############################
# Multiple Linear Regression
###############################
df = advertising.copy()

X = df.drop("sales", axis=1)
y = df[["sales"]]


#######################
# Model
#######################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)


reg_model = LinearRegression().fit(X_train, y_train)

# sabit katsayı
reg_model.intercept_[0]

# agirliklar
reg_model.coef_

# Asagadaki gözlem degerlerine göre satisin beklenen degeri nedir?
# TV: 30
# radio: 10
# newspaper: 40


new_data = [[30], [10], [40]]
new_data = pd.DataFrame(new_data).T

reg_model.predict(new_data)

#######################
# Tahmin Başarısınısın degerlendirme
#######################

# Train hatası - 1.7369025901470923
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Train r-kare - 0.8959372632325174
reg_model.score(X_train, y_train)

# Test hatası - 1.4113417558581582
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test r-kare - 0.8927605914615384 oranunda bağımsız değişkenlerin bağımlı değişkenleri açıklama yüzdesi
reg_model.score(X_test, y_test)


#############################################
# 10 katlı CV ile train hatası - 0.8959372632325174

np.mean(
    np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error"))
)

# 5 katlı CV ile train hatası - 1.7175247278732086
np.mean(
    np.sqrt(-cross_val_score(reg_model, X, y, cv=5, scoring="neg_mean_squared_error"))
)

#############################################
# Simple Linear Regression Gradient Descent from Scratch
#############################################


# Cost function - MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    
    b_deriv_sum = 0
    w_deriv_sum = 0
    
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        
        b_deriv_sum += y_hat - y
        w_deriv_sum += (y_hat - y) * X[i]
        
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print(
        "Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(
            initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)
        )
    )

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print(
        "After {0} iterations b = {1}, w = {2}, mse = {3}".format(
            num_iters, b, w, cost_function(Y, b, w, X)
        )
    )
    return cost_history, b, w


df = advertising.copy()

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
