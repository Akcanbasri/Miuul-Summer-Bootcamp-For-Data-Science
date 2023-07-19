import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = r"C:\Users\Lenova\Desktop\MiuulYazKampı\data\data.csv"
breast_cancer = pd.read_csv(file_path)
df = breast_cancer.copy()
df = df.iloc[:, 1:]
df.head()

num_cols = [col for col in df.columns if df[col].dtypes in [int, float]]

corr = df[num_cols].corr()

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(
    corr, annot=True, fmt=".2f", annot_kws={"size": 10}, linewidths=0.5, cmap="Blues"
)
plt.show()

# Yüksek korelasyonlu degiskenlerin silinmesi

corr_matrix = df.corr().abs()

upper_triangle_matrix = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# yüzde doksandan büyük korelasyonu olan degiskenlerin isimlerini alalım
drop_list = [
    col
    for col in upper_triangle_matrix.columns
    if any(upper_triangle_matrix[col] > 0.90)
]

corr_matrix[drop_list]

df.drop(drop_list, axis=1)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = r"C:\Users\Lenova\Desktop\MiuulYazKampı\data\data.csv"
breast_cancer = pd.read_csv(file_path)
df = breast_cancer.copy()
df = df.iloc[:, 1:]


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


num_cols = [col for col in df.columns if df[col].dtypes in [int, float]]

corr = df[num_cols].corr()

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(
    corr, annot=True, fmt=".2f", annot_kws={"size": 10}, linewidths=0.5, cmap="Blues"
)
plt.show()

# Yüksek korelasyonlu degiskenlerin silinmesi

corr_matrix = df.corr().abs()

upper_triangle_matrix = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# yüzde doksandan büyük korelasyonu olan degiskenlerin isimlerini alalım
drop_list = [
    col
    for col in upper_triangle_matrix.columns
    if any(upper_triangle_matrix[col] > 0.90)
]

corr_matrix[drop_list]

df.drop(drop_list, axis=1)

high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)

drop_list = high_correlated_cols(df.drop(drop_list, axis=1), plot=True)


