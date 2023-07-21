#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
persona = pd.read_csv("persona.csv")
df = persona.copy()
df.head()


# veri setin değişkenlerini türelerine göre ayırıyoruz.
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


# cat_th a 6 verdik çünkü price bir numeric degiskendir.
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=6)


# gelismis kesifci analizi fonsiyonu ile veri seti hakkında genel bilgileri gösteriyoruz.
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

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()  # 2

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()  # 6

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY").agg({"PRICE": "sum"}).sort_values("PRICE", ascending=False)

# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
df.groupby("SOURCE").agg({"PRICE": "count"}).sort_values("PRICE", ascending=False)

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.pivot_table(values="PRICE", index="SOURCE", aggfunc="mean")

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.pivot_table(values="PRICE", index="COUNTRY", columns="SOURCE", aggfunc="mean")

#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################
agg_df = df.pivot_table(
    values="PRICE", index=["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean"
)
#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
agg_df = agg_df.sort_values("PRICE", ascending=False)

#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
# agg_df.reset_index(inplace=True)
agg_df.reset_index(inplace=True)

#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'
age_labels = ["0-18", "19-23", "24-30", "31-40", "41-70"]
agg_df["age_group"] = pd.cut(
    agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=age_labels
)
agg_df.head()

#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

# Müşteri özelliklerinden customers_level_based'i oluşturuyoruz
agg_df["customers_level_based"] = (
    agg_df["COUNTRY"]
    + "_"
    + agg_df["SOURCE"]
    + "_"
    + agg_df["SEX"]
    + "_"
    + pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=age_labels).astype(str)
)

# customers_level_based değerlerini tekilleştiriyoruz
agg_df["customers_level_based"] = (
    agg_df.groupby("customers_level_based")["PRICE"]
    .transform("mean")
    .sort_values(ascending=False)
    .index
)


#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,

# PRICE'a göre segmentlere ayırma
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], q=4, labels=["D", "C", "B", "A"])

# Segmentleri betimleme
segment_summary = agg_df.groupby("SEGMENT").agg(
    {"PRICE": ["mean", "min", "max", "count"]}
)


#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?


# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
