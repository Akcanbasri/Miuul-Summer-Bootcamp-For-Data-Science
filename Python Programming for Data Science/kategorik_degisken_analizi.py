import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)

titanic = sns.load_dataset("titanic")
df = titanic.copy()
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["sex"].nunique()

df.info()

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

df["embarked"].value_counts()
# embarked değişkeninin sınıflarının frekansları yüzde olarak
100 * df["embarked"].value_counts() / len(df)

def cat_sum(data_frame, colm_name, plot=False):
    print(pd.DataFrame({colm_name: data_frame[colm_name].value_counts(),
                        "Ratio": 100 * data_frame[colm_name].value_counts() / len(data_frame)}))
    print("###############################################")
    
    if plot:
        sns.countplot(x=data_frame[colm_name], data=data_frame)
        plt.show(block=True)

for i in cat_cols:
    cat_sum(df, i)
    
# Part 2

cat_sum(df, "embarked", plot=True)

for i in cat_cols:
    if df[i].dtypes == "bool":
        # bool değişkenin adını yazdır
        print(i)
        print("----------------------------------------")
    else:
        cat_sum(df, i, plot=True)
        
df["adult_male"].astype(int)

for i in cat_cols:
    if df[i].dtypes == "bool":
        df[i] = df[i].astype(int)
        cat_sum(df, i, plot=True)
    else:
        cat_sum(df, i, plot=True)


# bu yol önerilmez
def cat_sum(data_frame, colm_name, plot=False):
    if data_frame[colm_name].dtypes == "bool":
        data_frame[colm_name] = data_frame[colm_name].astype(int)
        print(pd.DataFrame({colm_name: data_frame[colm_name].value_counts(),
                            "Ratio": 100 * data_frame[colm_name].value_counts() / len(data_frame)}))
        print("###############################################")
        
        if plot:
            sns.countplot(x=data_frame[colm_name], data=data_frame)
            plt.show(block=True)
    else:
        print(pd.DataFrame({colm_name: data_frame[colm_name].value_counts(),
                            "Ratio": 100 * data_frame[colm_name].value_counts() / len(data_frame)}))
        print("###############################################")
        
        if plot:
            sns.countplot(x=data_frame[colm_name], data=data_frame)
            plt.show(block=True)
            
cat_sum(df, "adult_male", plot=True)

# önerilen yol 

def cat_sum(data_frame, colm_name, plot=False):
    print(pd.DataFrame({colm_name: data_frame[colm_name].value_counts(),
                        "Ratio": 100 * data_frame[colm_name].value_counts() / len(data_frame)}))
    print("###############################################")
    
    if plot:
        sns.countplot(x=data_frame[colm_name], data=data_frame)
        plt.show(block=True)
        
for i in cat_cols:
    if df[i].dtypes == "bool":
        df[i] = df[i].astype(int)
        cat_sum(df, i, plot=True)
    else:
        cat_sum(df, i, plot=True)