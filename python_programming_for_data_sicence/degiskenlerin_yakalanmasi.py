import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.copy()
df.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """ we can use this function for getting categorical, numerical, categorical but cardinal variables names

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
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) 
                in ["object", "category", "bool"]]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() <  cat_th
                   and dataframe[col].dtypes in ["int64", "float64"]]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and str(dataframe[col].dtypes) in ["object", "category"]]
    
    cat_cols = cat_cols + num_but_cat
    
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes 
            in ["int64", "float64"] and col not in cat_cols]

    
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    
    
    return cat_cols, num_cols, cat_but_car 
        
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(data_frame, colm_name, plot=False):
    print("###############################################")
    print(pd.DataFrame({colm_name: data_frame[colm_name].value_counts(),
                        "Ratio": 100 * data_frame[colm_name].value_counts() / len(data_frame)}))
    
    if plot:
        sns.countplot(x=data_frame[colm_name], data=data_frame)
        plt.show(block=True)
    print("###############################################")
        
for i in cat_cols:
    # bool değişkeni int yapma
    if df[i].dtypes == "bool":
        df[i] = df[i].astype(int)
        cat_summary(df, i, plot=True)
    else:
        cat_summary(df, i, plot=True)
        
def num_summary(dataframe, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)
    
    if plot:
        dataframe[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show(block=True)
        
for col in num_cols:
    print("###############################################")
    num_summary(df, col, plot=True)
    print("###############################################")
    