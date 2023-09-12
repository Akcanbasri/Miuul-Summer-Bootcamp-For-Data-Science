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
    np.sqrt(
        -cross_val_score(
            reg_model,X, y, cv=10, scoring="neg_mean_squared_error"
        )
    )
)

# 5 katlı CV ile train hatası - 1.7175247278732086
np.mean(
    np.sqrt(
        -cross_val_score(
            reg_model,X, y, cv=5, scoring="neg_mean_squared_error"
        )
    )
)
 