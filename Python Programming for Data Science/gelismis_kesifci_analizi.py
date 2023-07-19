import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)

titanic = sns.load_dataset("titanic")
df = titanic.copy()
df.head()
df.tail()
df.shape
df.info()
df.index
df.columns
df.describe().T
df.isnull().values.any()
df.isnull().sum()   


def check_df(data_frame, head=5):
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
    
check_df(df)
    
    
tips = sns.load_dataset("tips")
check_df(tips)