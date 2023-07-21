"""
Görev 1: List Comprehension yapisi kullanarak car_crashes verisindeki numeric degiskenlerin isimlerini büyük
harfe çeviriniz ve basina NUM ekleyiniz. 

Beklenen Çikti:

['NUM TOTAL',
' NUM_SPEEDING',
'NUM ALCOHOL',
'NUM NOT DISTRACTED',
NUM NO PREVIOUS',
'NUM INS PREMIUM',
'NUM_INS LOSSES'
,ABBREV]
"""

import seaborn as sns
import pandas as pd

car_crashes = sns.load_dataset("car_crashes")
df = car_crashes.copy()
# List Comprehension for upper case
df.columns = [col.upper() for col in df.columns] 
# List Comprehension for numeric columns
only_numeric = ["NUM_"+ col for col in df.columns if df[col].dtype != "O"] 

""" 
Görev 2: List Comprehension yapısını kullanarak car_crashes verisinde isminde "no" barindirmayan
degiskenlerin isimlerinin sonuna "FLAG" yaziniz.
"""

# List Comprehension for "no" in column names
without_no = [col + "_FLAG" for col in df.columns if "no" not in col.lower()]

""" 
Görev 3: List Comprehension yapisi kullanarak asagida verilen degisken isimlerinden FARKLI olan
degiskenlerin isimlerini seçiniz ve yeni bir dataframe olusturunuz.
"""

og_list = ["abbrev", "no_previous"]

# List Comprehension for deleting og_list from df
new_list = [col for col in df.columns if col not in og_list]

# create new dataframe
new_df = df[new_list]