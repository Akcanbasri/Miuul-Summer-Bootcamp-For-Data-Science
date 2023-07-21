import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")
df = titanic.copy()
df.head()

# Kadın ve erkek yolcu sayısını bulunuz.
df["sex"].value_counts()

# her bir sutuna ait unique degerlerin saysını bulunuz.
df.nunique()

# pclass değişkeninin unique değerlerinin sayısını bulunuz.
df["pclass"].nunique()

# pclass ve parch degiskenlerinin unique degerlerinin sayisini bulunuz.
df[["pclass", "parch"]].nunique()

# mbarked degiskeninin tipini kontrol ediniz. Tipini category olarak degistiriniz ve tekrar kontrol ediniz.
df["embarked"].dtypes
df["embarked"] = df["embarked"].astype("category")
df.info()

# embarked degeri C olanlarin tüm bilgelerini gösteriniz.
df["embarked"].value_counts()  # C: 168
df[df["embarked"] == "C"]

# embarked degeri S olmayanlarin tüm bilgelerini gösteriniz.
df["embarked"].value_counts()  # S: 644, S olmayan: 247
df[df["embarked"] != "S"]

# Yasi 30 dan küçük ve kadin olan yolcularin tüm bilgilerini gösteriniz.
df[(df["sex"] == "female") & (df["age"] < 30)]

# Fare' 500'den büyük veya yasi 70 den büyük yolcularin bilgilerini gösterini:
df[(df["fare"] > 500) | (df["age"] > 70)]

# Her bir degiskendeki bos degerlerin toplamini bulunuz.
df.isnull().sum()

# who degiskenini dataframe' den gikariniz.
df.drop("who", axis=1, inplace=True)

# deck degikenindeki bos degerleri deck degiskenin en ok tekrar eden degeri (mode) ile doldurunuz.
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df.isnull().sum()

# 30 yasin altinda olanlar 1, 30 a esit ve üstünde olanlara 0 vericek bir fonksiyon yazin. Yazdiginiz fonksiyonu kullanarak titanik veri
# setinde age_flag adinda bir degisken olusturunuz olusturunuz. (apply ve lambda yapilarini kullaniniz)


def age_flag(x):
    if x < 30:
        return 1
    else:
        return 0


df["age_flag"] = df["age"].apply(lambda x: age_flag(x))

# tips veri setini tanumlayınız.

tips = sns.load_dataset("tips")
df = tips.copy()
df.head()
# Time degiskeninin kategorilerine (Dinner, Lunch) göre total_bill degerlerinin toplamini, min, max ve ortalamasini bulunuz.
df.groupby("time")["total_bill"].agg(["sum", "min", "max", "mean"])

# Günlere ve time göre total_bill degerlerinin toplamini, min, max ve ortalamasini bulunuz.
df.groupby(["day", "time"])["total_bill"].agg(["sum", "min", "max", "mean"])

# Lunch zamanina ve kadin müsterilere ait total_bill ve tip degerlerinin day' göre toplamini, min, max ve ortalamasini bulunuz.
# kadın ve lunch olanları filtrele
df.loc[(df["sex"] == "Female") & (df["time"] == "Lunch")].groupby("day")[
    "total_bill", "tip"
].agg(["sum", "min", "max", "mean"])

# size'¡ 3'ten küçük, total_bill'i 10'dan büyük olan siparislerin ortalamasi nedir? (loc kullaniniz)
df.loc[(df["size"] < 3) & (df["total_bill"] > 10)].mean()

# total_bill_tip_sum adinda yeni bir degisken olusturunuz. Her bir müsterinin ödedigi totalbill ve tip in toplamini versin.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

# total_bill_tip_sum degiskenine göre büyükten küçüge siralayiniz ve ilk 30 kisiyi yeni bir dataframe'e atayiniz.

df_new = df.sort_values("total_bill_tip_sum", ascending=False).head(30)