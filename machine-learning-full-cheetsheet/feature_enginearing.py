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


#####################################################################################
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculates the lower and upper limits for outliers based on the interquartile range method.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the column to calculate the outlier limits for.
    col_name (str): The name of the column to calculate the outlier limits for.
    q1 (float, optional): The percentile value for the first quartile. Default is 0.25.
    q3 (float, optional): The percentile value for the third quartile. Default is 0.75.

    Returns:
    tuple: A tuple containing the lower and upper limits for outliers.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


#####################################################################################
def check_outlier(dataframe, col_name):
    """
    Check if there are any outliers in a given column of a dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to check for outliers.
    col_name (str): The name of the column to check for outliers.

    Returns:
    bool: True if there are outliers, False otherwise.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ].any(axis=None):
        return True
    else:
        return False


#####################################################################################
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


#####################################################################################
def grab_outliers(dataframe, col_name, index=False):
    """
    This function takes a dataframe and a column name as input and returns the outliers of the column.
    If index is True, it also returns the indices of the outliers.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    col_name (str): The name of the column to find outliers for.
    index (bool, optional): Whether to return the indices of the outliers. Defaults to False.

    Returns:
    pandas.DataFrame or pandas.Index: The outliers of the column. If index is True, returns the indices of the outliers.
    """
    low, up = outlier_thresholds(dataframe, col_name)

    if (
        dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0]
        > 10
    ):
        print(
            dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head()
        )
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[
            ((dataframe[col_name] < low) | (dataframe[col_name] > up))
        ].index
        return outlier_index


# fonksiyon kullanımı
# grab_outliers(df, "Age", True)


#####################################################################################
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[
        ~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))
    ]
    return df_without_outliers


# fonksiyon kullanımı
# cat_cols, num_cols, cat_but_car = grab_col_names(df)

# num_cols = [col for col in num_cols if col not in "PassengerId"]

# df.shape

# for col in num_cols:
#     new_df = remove_outlier(df, col)

# df.shape[0] - new_df.shape[0]


#####################################################################################
def replace_with_thresholds(dataframe, variable):
    """
    Replaces the values in a given variable of a dataframe with the lower and upper limits of the variable's outliers.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the variable to be replaced.
    variable (str): The name of the variable to be replaced.

    Returns:
    None
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# fonksiyon kullanımı
# for col in num_cols:
#     replace_with_thresholds(df, col)


#####################################################################################
def missing_values_table(dataframe, na_name=False):
    """
    This function takes a pandas dataframe as input and returns a table of missing values for each column in the dataframe.
    If na_name is True, it also returns a list of column names that contain missing values.
    """
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


# missing_values_table(df, True)


# 1. solutuion
# df.dropna().shape
#####################################################################################
# 2. solution
# df["Age"].fillna(df["Age"].mean()).isnull().sum()
# df["Age"].fillna(df["Age"].median()).isnull().sum()
# df["Age"].fillna(0).isnull().sum()

# # df.apply(lambda x: x.fillna(x.mean()), axis=0)

# df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

# dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

# dff.isnull().sum().sort_values(ascending=False)

# df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

# df["Embarked"].fillna("missing")

# df.apply(
#     lambda x: x.fillna(x.mode()[0])
#     if (x.dtype == "O" and len(x.unique()) <= 10)
#     else x,
#     axis=0,
# ).isnull().sum()

# ###################
# # Kategorik Değişken Kırılımında Değer Atama
# ###################


# df.groupby("Sex")["Age"].mean()

# df["Age"].mean()

# df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# df.groupby("Sex")["Age"].mean()["female"]

# df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")[
#     "Age"
# ].mean()["female"]

# df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")[
#     "Age"
# ].mean()["male"]

# df.isnull().sum()

#####################################################################################

# cat_cols, num_cols, cat_but_car = grab_col_names(df)
# num_cols = [col for col in num_cols if col not in "PassengerId"]
# dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

# dff.head()

# # değişkenlerin standartlatırılması
# scaler = MinMaxScaler()
# dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
# dff.head()


# # knn'in uygulanması.
# from sklearn.impute import KNNImputer

# imputer = KNNImputer(n_neighbors=5)
# dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
# dff.head()

# dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

# df["age_imputed_knn"] = dff[["Age"]]

# df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
# df.loc[df["Age"].isnull()]


#####################################################################################
# missing_values_table(df, True)
# na_cols = missing_values_table(df, True)
import pandas as pd
import numpy as np

def missing_vs_target(dataframe, target, na_columns):
    """
    This function takes a dataframe, target column name and a list of columns with missing values as input.
    It creates a new column for each column in na_columns with "_NA_FLAG" suffix to indicate if the value is missing or not.
    Then, it groups the dataframe by each "_NA_FLAG" column and calculates the mean of the target column and the count of each group.
    Finally, it prints the result for each group.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    target (str): The name of the target column.
    na_columns (list): A list of column names with missing values.

    Returns:
    None
    """
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


# missing_vs_target(df, "Survived", na_cols)

#####################################################################################
# df = load()
# na_cols = missing_values_table(df, True)
# # sayısal değişkenleri direk median ile oldurma
# df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# # kategorik değişkenleri mode ile doldurma
# df.apply(
#     lambda x: x.fillna(x.mode()[0])
#     if (x.dtype == "O" and len(x.unique()) <= 10)
#     else x,
#     axis=0,
# ).isnull().sum()
# # kategorik değişken kırılımında sayısal değişkenleri doldurmak
# df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# # Tahmine Dayalı Atama ile Doldurma
# missing_vs_target(df, "Survived", na_cols)


#####################################################################################
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

def label_encoder(dataframe: DataFrame, binary_col: str) -> DataFrame:
    """
    Encodes binary categorical features using LabelEncoder.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    binary_col (str): The name of the binary column to be encoded.

    Returns:
    pandas.DataFrame: The encoded dataframe.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# binary_cols = [
#     col
#     for col in df.columns
#     if df[col].dtype not in [int, float] and df[col].nunique() == 2
# ]
# for ile bu listede dön


#####################################################################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """
    One-hot encode categorical columns in a pandas dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be encoded.
    categorical_cols (list): A list of column names to be one-hot encoded.
    drop_first (bool): Whether to drop the first column of each encoded category. Default is True.

    Returns:
    pandas.DataFrame: The encoded dataframe.
    """
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first
    )
    return dataframe


# ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
# one_hot_encoder(df, ohe_cols).head()

#####################################################################################
# Rare Encoding
#############################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

# df = load_application_train()
# df["NAME_EDUCATION_TYPE"].value_counts()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    """
    This function takes a dataframe and a column name as input and returns a summary of the column's value counts and ratios.
    If plot is set to True, it also displays a countplot of the column's values.
    
    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    col_name (str): The name of the column to summarize.
    plot (bool): Whether to display a countplot of the column's values. Default is False.
    
    Returns:
    None
    """
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


# for col in cat_cols:
#     cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

# df["NAME_INCOME_TYPE"].value_counts()

# df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(dataframe, target, cat_cols):
    """
    Analyzes the categorical columns in a given dataframe and prints the count, ratio, and target mean for each unique value in the column.

    Args:
    - dataframe: pandas DataFrame
    - target: str, name of the target column
    - cat_cols: list of str, names of the categorical columns to analyze

    Returns:
    - None
    """
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


# rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################


import pandas as pd
import numpy as np

def rare_encoder(dataframe: pd.DataFrame, rare_perc: float) -> pd.DataFrame:
    """
    Replaces infrequent categorical values in a dataframe with the string 'Rare'.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe.
    rare_perc (float): The threshold percentage below which a categorical value is considered rare.

    Returns:
    pd.DataFrame: A copy of the input dataframe with rare categorical values replaced with 'Rare'.
    """
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


# new_df = rare_encoder(df, 0.01)

# rare_analyser(new_df, "TARGET", cat_cols)

# df["OCCUPATION_TYPE"].value_counts()

#####################################################################################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

# df = load()
# ss = StandardScaler()
# df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
# df.head()


###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

# rs = RobustScaler()
# df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
# df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

# mms = MinMaxScaler()
# df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
# df.describe().T

# df.head()

# age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, numerical_col, plot=False):
    """
    This function takes a pandas dataframe and a numerical column as input and returns a summary of the column's statistics.
    It also has an optional plot parameter that, if set to True, will display a histogram of the column's values.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the numerical column.
    numerical_col (str): The name of the numerical column to summarize.
    plot (bool, optional): Whether to display a histogram of the column's values. Defaults to False.

    Returns:
    None
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# for col in age_cols:
#     num_summary(df, col, plot=True)


#####################################################################################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

# df["Age_qcut"] = pd.qcut(df["Age"], 5)


#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

# df = load()
# ss = StandardScaler()
# df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
# df.head()


###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

# rs = RobustScaler()
# df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
# df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

# mms = MinMaxScaler()
# df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
# df.describe().T

# df.head()

# age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, numerical_col, plot=False):
    """
    This function takes a pandas dataframe and a numerical column as input and returns a summary of the column's statistics.
    It also has an optional plot parameter that, if set to True, will display a histogram of the column's values.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    numerical_col (str): The name of the numerical column to summarize.
    plot (bool, optional): Whether to display a histogram of the column's values. Defaults to False.

    Returns:
    None
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# for col in age_cols:
#     num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

# df["Age_qcut"] = pd.qcut(df["Age"], 5)


#####################################################################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

# df = load()
# df.head()

# df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")

# df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})


# from statsmodels.stats.proportion import proportions_ztest

# test_stat, pvalue = proportions_ztest(
#     count=[
#         df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
#         df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum(),
#     ],
#     nobs=[
#         df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
#         df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0],
#     ],
# )

# print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))


# df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
# df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

# df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


# test_stat, pvalue = proportions_ztest(
#     count=[
#         df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
#         df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum(),
#     ],
#     nobs=[
#         df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
#         df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0],
#     ],
# )

# print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

# df.head()

###################
# Letter Count
###################

# df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

# df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################

# df["NEW_NAME_DR"] = df["Name"].apply(
#     lambda x: len([x for x in x.split() if x.startswith("Dr")])
# )

# df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

###################
# Regex ile Değişken Türetmek
###################

# df.head()

# df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)


# df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg(
#     {"Survived": "mean", "Age": ["count", "mean"]}
# )

#############################################
# Date Değişkenleri Üretmek
#############################################

# dff = pd.read_csv("datasets/course_reviews.csv")
# dff.head()
# dff.info()

# dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# # year
# dff["year"] = dff["Timestamp"].dt.year

# # month
# dff["month"] = dff["Timestamp"].dt.month

# # year diff
# dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

# # month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
# dff["month_diff"] = (
#     (date.today().year - dff["Timestamp"].dt.year) * 12
#     + date.today().month
#     - dff["Timestamp"].dt.month
# )


# # day name
# dff["day_name"] = dff["Timestamp"].dt.day_name()

# dff.head()

# date


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
# df = load()
# df.head()

# df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

# df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

# df.loc[(df["SEX"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

# df.loc[
#     (df["SEX"] == "male") & (df["Age"] > 21) & (df["Age"] < 50), "NEW_SEX_CAT"
# ] = "maturemale"

# df.loc[(df["SEX"] == "male") & (df["Age"] >= 50), "NEW_SEX_CAT"] = "seniormale"

# df.loc[(df["SEX"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"

# df.loc[
#     (df["SEX"] == "female") & (df["Age"] > 21) & (df["Age"] < 50), "NEW_SEX_CAT"
# ] = "maturefemale"

# df.loc[(df["SEX"] == "female") & (df["Age"] >= 50), "NEW_SEX_CAT"] = "seniorfemale"


# df.head()

# df.groupby("NEW_SEX_CAT")["Survived"].mean()


#####################################################################################
# model değişkeni değerlendirme
# y = df["SURVIVED"]
# X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.30, random_state=17
# )

# from sklearn.ensemble import RandomForestClassifier

# rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
# y_pred = rf_model.predict(X_test)
# accuracy_score(y_pred, y_test)
# num=len(X) üsstten gelecek

def plot_importance(model, features, num=len(), save=False):
    """
    Plots the feature importances of a given model.

    Args:
        model: A trained machine learning model.
        features: A pandas DataFrame containing the features used to train the model.
        num: An integer representing the number of features to display in the plot. Default is the number of features in the DataFrame.
        save: A boolean indicating whether to save the plot as a PNG file. Default is False.

    Returns:
        None
    """
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features.columns}
    )
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")
        
# plot_importance(rf_model, X_train)