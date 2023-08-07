import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# veri setine genel bakış
def check_df(data_frame, head=5):
    """
    This function takes a pandas dataframe and prints out its shape, data types, head, tail, number of missing values, and descriptive statistics.

    Parameters:
    data_frame (pandas.DataFrame): The dataframe to be checked.
    head (int): The number of rows to be displayed for head and tail. Default is 5.
    """
    print("##################### Shape #####################")
    print(data_frame.shape)
    print("##################### Types #####################")
    print(data_frame.dtypes)
    print("##################### Head #####################")
    print(data_frame.head(head))
    print("##################### Tail #####################")
    print(data_frame.tail(head))
    print("##################### NA #####################")
    print(data_frame.isnull().sum())
    print("##################### Describe #####################")
    print(data_frame.describe().T)


# veriyi türlerine göre ayırma fonksiyonu
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """we can use this function for getting categorical, numerical, categorical but cardinal variables names

    Args:
    -------
        dataframe (pandas.DataFrame): all data
        cat_th ([int, floot], optional):

        numeric fakat kategorik olan değişkenler için sınıf eşiği. Defaults to 10.

        car_th ([int, floot], optional):
        katagorik fakat kardinal değişkenler için sınıf eşik değeri. Defaults to 20.

    Returns:
    -------
    cat_cols: List
        kategorik değişken isimleri

    num_cols: List
        numerik değişken isimleri

    cat_but_car: List
        kategorik görünüp aslında kardinal olan değişken isimleri

    Notes:
    -------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat  cat_cols un içinde
        return olan üç liste toplamı toplam değişken sayısına eşittir.

    """

    # cat_cols, cat_but_car
    cat_cols = [
        col
        for col in dataframe.columns
        if str(dataframe[col].dtypes) in ["object", "category", "bool"]
    ]

    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th
        and dataframe[col].dtypes in ["int64", "float64"]
    ]

    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th
        and str(dataframe[col].dtypes) in ["object", "category"]
    ]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [
        col
        for col in dataframe.columns
        if dataframe[col].dtypes in ["int64", "float64"] and col not in cat_cols
    ]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


# cat cols için özet istatistikler
def cat_summary(data_frame, colm_name, plot=False):
    """
    This function takes a pandas dataframe and a column name as input and returns a summary of the column's value counts
    and their ratios. If the plot parameter is set to True, it also displays a countplot of the column's values.

    Parameters:
    data_frame (pandas.DataFrame): The input dataframe
    colm_name (str): The name of the column to be summarized
    plot (bool): Whether to display a countplot of the column's values (default is False)

    Returns:
    None
    """
    print("###############################################")
    print(
        pd.DataFrame(
            {
                colm_name: data_frame[colm_name].value_counts(),
                "Ratio": 100 * data_frame[colm_name].value_counts() / len(data_frame),
            }
        )
    )

    if plot:
        sns.countplot(x=data_frame[colm_name], data=data_frame)
        plt.show(block=True)
    print("###############################################")


# cat cols için bool değişkenlerin int yapılması gerekiyor
# for i in cat_cols:
#     # bool değişkeni int yapma
#     if df[i].dtypes == "bool":
#         df[i] = df[i].astype(int)
#         cat_summary(df, i, plot=True)
#     else:
#         cat_summary(df, i, plot=True)


# num cols için özet istatistikler
def num_summary(dataframe, num_cols, plot=False):
    """
    This function takes a pandas dataframe and a list of numerical column names as input and returns a summary of the
    numerical columns including count, mean, standard deviation, minimum, 5th percentile, 10th percentile, 20th percentile,
    30th percentile, 40th percentile, 50th percentile (median), 60th percentile, 70th percentile, 80th percentile, 90th
    percentile, 95th percentile, 99th percentile and maximum. If plot is set to True, it also displays a histogram of
    the numerical columns.

    Parameters:
    dataframe (pandas.DataFrame): The pandas dataframe to be analyzed.
    num_cols (list): A list of numerical column names to be analyzed.
    plot (bool, optional): Whether to display a histogram of the numerical columns. Defaults to False.

    Returns:
    None
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)

    if plot:
        dataframe[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show(block=True)


# num cols için çağırma
# for col in num_cols:
#     print("###############################################")
#     num_summary(df, col, plot=True)
#     print("###############################################")


# target analizi için özet istatistikler cat_cols
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    This function takes in a dataframe, target column name and a categorical column name as input.
    It groups the dataframe by the categorical column and calculates the mean of the target column for each group.
    It then returns a pandas dataframe with the mean target values for each group.
    """
    print(
        pd.DataFrame(
            {"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}
        ),
        end="\n\n",
    )


# target analizi için özet istatistikler num_cols
def target_summary_with_num(dataframe, target, numerical_col):
    """
    This function groups the given dataframe by the target column and calculates the mean of the numerical column for each group.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    target (str): The name of the target column.
    numerical_col (str): The name of the numerical column.

    Returns:
    None
    """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")


# yüksel korelasyonlu değişkenlerin tespiti
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    This function takes a pandas dataframe as input and returns a list of column names that have a correlation
    coefficient greater than the specified threshold. If plot is set to True, it also displays a heatmap of the
    correlation matrix using seaborn and matplotlib.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    plot (bool): Whether or not to display a heatmap of the correlation matrix. Default is False.
    corr_th (float): The correlation threshold. Columns with a correlation coefficient greater than this value
        will be included in the returned list. Default is 0.90.

    Returns:
    list: A list of column names that have a correlation coefficient greater than the specified threshold.
    """
    corr = dataframe.corr()
    corr_matrix = corr.abs()

    upper_triangle_matrix = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    drop_list = [
        col
        for col in upper_triangle_matrix.columns
        if any(upper_triangle_matrix[col] > corr_th)
    ]

    if plot:
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()

    return drop_list