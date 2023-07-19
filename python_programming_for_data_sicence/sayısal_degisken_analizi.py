import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")
df = titanic.copy()
df.head()

df[["age", "fare"]].describe().T

# list comprehension ile kategorik değişkenlerin isimlerini almak
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]
# list comprehension ile kategorik değişkenlerin isimlerini almak
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
# list comprehension ile kategorik değişkenlerin isimlerini almak
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]
df[cat_cols].nunique()
# numeric değişkenlerin isimlerini almak
[col for col in df.columns if col not in cat_cols]

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] 
            and col not in cat_cols]

def sum_num(dataframe, num_cols):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)
    
sum_num(df, "age")

for col in num_cols:
    sum_num(df, col)
    print("###############################################")
    
def sum_num(dataframe, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)
    
    if plot:
        dataframe[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show(block=True)
        
for col in num_cols:
    sum_num(df, col, plot=True)
    print("###############################################")