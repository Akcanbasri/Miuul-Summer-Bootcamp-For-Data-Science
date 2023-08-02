#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)


def load_application_train():
    data = pd.read_csv("./datasets/application_train.csv")
    return data


df = load_application_train()
df.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()


#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler
###################

sns.boxplot(x=df["Age"])
plt.show()

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)]

df[(df["Age"] < low) | (df["Age"] > up)].index

###################
# Aykırı Değer Var mı Yok mu?
###################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)
df[(df["Age"] < low)].any(axis=None)

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()


df[(df["Fare"] < low) | (df["Fare"] > up)].index


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "Age")
check_outlier(df, "Fare")

###################
# grab_col_names
###################

dff = load_application_train()
dff.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))


# Aykırı değerlerin kendine erişmek
def grab_outlier(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if len(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]) > 10:
        print(
            dataframe[((dataframe[col_name] < low)) | (dataframe[col_name] > up)].head()
        )
    else:
        dataframe[((dataframe[col_name] < low)) | (dataframe[col_name] > up)]

    if index:
        outlier_index = dataframe[
            ((dataframe[col_name] < low)) | (dataframe[col_name] > up)
        ].index
        return outlier_index


age_index = grab_outlier(df, "Age", True)


outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outlier(df, "Age", True)

# Aykırı Değerleri silme

low, up = outlier_thresholds(df, "Fare")

df.shape  # 891 gözlem


df[~((df["Fare"] < low) | (df["Fare"] > up))].shape


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outlier = dataframe[
        ~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))
    ]

    return df_without_outlier


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape  # 891 gözlem var

for col in num_cols:
    new_df = remove_outlier(df, col)

new_df.shape  # 775 gözleme düştü

# !!!!!! bir numeric değer varsa o satır tamamen siliniyor.


########################################################
# Baskılama yöntemi(re-assignment with threashold)
########################################################

low, up = outlier_thresholds(df, "Fare")


df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up
df.loc[(df["Fare"] < low), "Fare"] = low


def replaace_with_threasholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit


df = load()
df.shape
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))  # True

# sınırlarla outlier değerleri dolduruyoruz.
for col in num_cols:
    replaace_with_threasholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))  # False


#####################################################
# Recap
#####################################################

df = load()
outlier_thresholds(df, "Age")
check_outlier(df, "Age")  # True yani outlier var demek
grab_outlier(df, "Age", True)

remove_outlier(df, "Age").shape  # outlier değerleri direk çıkarır
replaace_with_threasholds(df, "Age")  # sınır değerleri ouytlier üzerine yazar.

check_outlier(df, "Age")  # False Outlier değerler yok


#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################

# 17, 3

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()
df.shape
for col in df.columns:
    print(col, check_outlier(df, col))


low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style=".-")
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th]
df[df_scores < th].shape

# neden bu üçü aykırı anlamak istiyorum
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index  # aykırı degerlerin iindexlerini yakalama

# drop ile atıyoruz değerleri
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

df = load()
df.head()

# veri setinin içinde herhangi bir yerde eksik değer varsa
df.isnull().values.any()  # True

# degiskendeki eksik değişkenlerin toplamını bulacağız
df.isnull().values.sum()  # eksik deger sayısı

# hangi degiskenlerde kaç tane var ona bakarız
df.isnull().sum()

# null olmayan kaç değer vardır ?
df[df.notnull().all(axis=1)]

df.isnull().sum().sort_values(ascending=False)

# yüzde ile veri setinin eksik değerlerinin yönetilmesi
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# null column isimlerini yakalanması
na_col = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (
        dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100
    ).sort_values(ascending=False)
    missing_df = pd.concat(
        [n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"]
    )
    print(missing_df, end="\n")
    if na_name:
        return na_columns


missing_values_table(df)
missing_values_table(
    df, True
)  # sutun listesine ulaşmak için na_name true yapmamız lazım.


#######################################################
# Eksik Deger problemini çözme
#######################################################
df = load()
df.head()

missing_values_table(df)

# çözüm 1 'Hızlıca silmek'
df.dropna().shape


# çözüm 2: 'Basit atama yöntemleriyle doldurmak'
df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].mean()).isnull().sum()  # 0 null


df["Age"].fillna(df["Age"].median())
df["Age"].fillna(df["Age"].median()).isnull().sum()  # 0 null

df["Age"].fillna(0)
df["Age"].fillna(0).isnull().sum()  # 0 null


# apply ve lambda ile null değer doldurma

df.apply(lambda x: x.fillna(x.mean() if x.dtypes != "O" else x), axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean() if x.dtypes != "O" else x), axis=0)

dff.isnull().sum().sort_values(ascending=False)

##############################################################
# Kategorik degiskenlerin null değerlerini inceleme ve doldurma
##############################################################

df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"].fillna(
    df["Embarked"].mode()[0]
).isnull().sum()  # 0 null 'S ile doldurduk'


df["Embarked"].fillna("missing")

# apply lambda for categoric columns
df.apply(
    lambda x: x.fillna(
        x.mode()[0] if (x.dtype == "O" and len(x.unique()) <= 10) else x
    ),
    axis=0,
)

df.apply(
    lambda x: x.fillna(
        x.mode()[0] if (x.dtype == "O" and len(x.unique()) <= 10) else x
    ),
    axis=0,
).isnull().sum().sort_values(ascending=False)

###################
# Kategorik Değişken Kırılımında Değer Atama
###################

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")[
    "Age"
].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")[
    "Age"
].mean()["male"]

df.isnull().sum()


#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################
df = load()
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

# Standartlaştırma işlemi
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# KNN algoritmasının uygulanması
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

# tahmine dayalı atama işlemi en yakın beş komşuya göre
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# min max scaler i tersine çeviriyoruz
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

# ilk df e atanmış değerleri ekliyoruz karşılaştırma için
df["age_inputed_knn"] = dff[["Age"]]

# yaş değişkeni için null değerleri ve atadığımız değerleri gösteriyoruz
df.loc[df["Age"].isnull(), ["Age", "age_inputed_knn"]]

############################
# Recap
############################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(
    lambda x: x.fillna(x.mode()[0])
    if (x.dtype == "O" and len(x.unique()) <= 10)
    else x,
    axis=0,
).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma

#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(
            pd.DataFrame(
                {
                    "TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                    "Count": temp_df.groupby(col)[target].count(),
                }
            ),
            end="\n\n\n",
        )


missing_vs_target(df, "Survived", na_cols)


###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(
    lambda x: x.fillna(x.mode()[0])
    if (x.dtype == "O" and len(x.unique()) <= 10)
    else x,
    axis=0,
).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)

#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

#############################################
# Label Encoding & Binary Encoding
#############################################

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0, 1])


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()

binary_cols = [
    col
    for col in df.columns
    if df[col].dtype not in [int, float] and df[col].nunique() == 2
]

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [
    col
    for col in df.columns
    if df[col].dtype not in [int, float] and df[col].nunique() == 2
]

df[binary_cols].head()


for col in binary_cols:
    label_encoder(df, col)


df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

########################################
# One Hot Encoding
########################################

df = load()
df.head()

df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(
    df, columns=["Embarked"], drop_first=True, dummy_na=True
).head()  # ilk sınıfı düşürdük birbiri üzerinden çoğalmaması için.


pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first
    )
    return dataframe


df = load()
# cat_cols, num_cols, cat_but_car  = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols, drop_first=True).head()

########################################
# Rare Encoding
########################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_INCOME_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(
        pd.DataFrame(
            {
                col_name: dataframe[col_name].value_counts(),
                "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
            }
        )
    )
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyzer(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(
            pd.DataFrame(
                {
                    "COUNT": dataframe[col].value_counts(),
                    "RATIO": dataframe[col].value_counts() / len(dataframe),
                    "TARGET_MEAN": dataframe.groupby(col)[target].mean(),
                }
            ),
            end="\n\n\n",
        )


rare_analyzer(df, "TARGET", cat_cols)


#############################################
# 3. Rare encoder'ın yazılması.
#############################################


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [
        col
        for col in temp_df.columns
        if temp_df[col].dtypes == "O"
        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)
    ]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)

rare_analyzer(new_df, "TARGET", cat_cols)

#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

df = load()
ss = StandardScaler()

df["Age_SScaler"] = ss.fit_transform(df[["Age"]])
df.describe().T

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df["Age_RScaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_MMScaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in age_cols:
    num_summary(df, col, plot=True)


###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df["Age"], 5)


#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()


df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

# Oran testi yapalım
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(
    count=[
        df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
        df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum(),
    ],
    nobs=[
        df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
        df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0],
    ],
)

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))


df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(
    count=[
        df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
        df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum(),
    ],
    nobs=[
        df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
        df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0],
    ],
)

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))