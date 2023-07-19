import pandas as pd
import numpy as np
import seaborn as sns


s = pd.Series([1,2,3,4,5])
s.index
s.dtype
s.size
s.ndim
s.values

titanic = sns.load_dataset("titanic")
df = titanic.copy()
df.head()
df.tail()
df.shape
df.info()
df.columns
df.describe().T
df.index
df.isnull().values.any()
df.isnull().sum()
df["sex"].value_counts()

df.index
df[0:13]

df.drop(0, axis = 0).head()

drop_list = [2,3,4,5]
df.drop(drop_list, axis = 0).head()

df.drop(drop_list, axis = 0, inplace = True)

# değişkeni indexe çevirme

df["age"].head()

df.age.head()

df.index = df["age"]

df.drop(["age"], axis = 1).head()
df.drop(["age"], axis = 1, inplace = True)

# indexi değişkene çevirme
df.index

df["age"] = df.index

#ikinci yol 
df.reset_index().head()
df = df.reset_index().head()

"age" in df

df["age"].head() # type: pandas.core.series.Series

df[['age']].head() # type: pandas.core.frame.DataFrame

df[["age", "alive"]].head()

colnames = ["age", "alive", "adult_male"]

df[colnames].head()

df["age2"] = df["age"] ** 2
df["age3"] = df["age"] / df["age2"]

df.drop("age3", axis =1, inplace = True)


df.drop(colnames, axis = 1).head()

df.loc[:, ~df.columns.str.contains("age")]

df = titanic.copy()
df.head()

#iloc integer location

df.iloc[0:3]

df.iloc[0, 0]

# loc label based indexing
df.loc[0:3]

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]
df.loc[0:3, ["age","who"]]

col_names = ["age", "embarked", "alive"]

df.loc[0:3, col_names] #type: pandas.core.frame.DataFrame

# koşullu seçimler

df[df["age"] > 50].head() # type: pandas.core.frame.DataFrame
df[df["age"] > 50].count()
df[df["age"] > 50]["age"].count()


df.loc[df["age"] > 50, ["age", "class"]].head()

# birdn fazla koşul için & kullanılır ve parantez kullanılır
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df_new = df.loc[(df["age"] > 50) 
       & (df["sex"] == "male") 
       & ((df["embark_town"]== "Southampton") | (df["embark_town"]== "Cherbourg")), 
       ["age", "class", "embark_town"]].head()

df["embark_town"].value_counts()
df_new["embark_town"].value_counts()

# Toplulaştırma ve Gruplama İşlemleri

df["age"].mean()

df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age":"mean"})

df.groupby("sex").agg({"age":["mean", "max", "min"],
                       "fare":["mean", "sum", "max"]})

df.groupby(["sex", "embark_town", "class"]).agg({"age":["mean"],
                       "survived":["mean"]})

df.head()

df.groupby(["sex", "embark_town", "class"]).agg({"age":["mean"],
                       "survived":["mean"],
                       "sex":["count"]})

# pivot table

df = titanic.copy()
df.head()

df.pivot_table("survived", "sex", ["embarked", "class"])

# cut fonksiyonu ile kategorik değişken oluşturma
# qcut ile eşit aralıklı kategorik değişken oluşturma
df["new_age"] = pd.cut(df["age"], [0,10,18,25,40,90])
df.head() 

df.pivot_table("survived", ["sex", "new_age"], ["embarked", "class"])

df.pivot_table("survived", ["sex"], ["new_age", "class"])

# apply ve lamda fonksiyonu

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5
df.head()

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

df.drop("new_age", axis = 1, inplace = True)

for col in df.columns:
    if "age" in col:
        print(col)
        

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())
        

for col in df.columns:
    if "age" in col:
        df["col"] = df[col]/10
        
df.head()

df[["age", "age2","age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: ((x - x.mean())- x.std())).head()


def standart_scaler(col):
    return (col - col.mean()) / col.std()

# df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()
df.head()

# concat ile dataframe birleştirme

import numpy as np

m = np.random.randint(1,30, size = (5,3))

df1 = pd.DataFrame(m, columns = ["var1", "var2", "var3"])
df2 = df1 + 99

df3 = pd.concat([df1, df2])
df3 = pd.concat([df1, df2], ignore_index = True) #ignore_index = True indexleri sıfırlar

df4 = pd.concat([df1, df2], axis = 1) # axis = 1 kolon bazında birleştirme yapar

# merge ile dataframe birleştirme

df1 = pd.DataFrame({'employees': ['Ali', 'Veli', 'Ayse', 'Fatma'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})

df2 = pd.DataFrame({'employees': ['Ayse', 'Ali', 'Veli', 'Fatma'],
                    "start_date": [2010, 2009, 2014, 2019]})

df4 = pd.DataFrame({'group': ["Accounting", "Engineering", "HR"],
                    "manager": ["Caner", "Mustafa", "Berkay"]})

df3 = pd.merge(df1, df2) # inner join yapar
df3 = pd.merge(df1, df2, on="employees") # inner join yapar

df3 = pd.merge(df3, df4,) # left join yapar

