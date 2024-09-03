
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("car_crashes")

df.columns
df.info()
df.head()

#  List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini
#  büyük harfe çeviriniz ve başına NUM ekleyiniz ve df te kalıcı hale getir

df = ["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

###
#List Comprehension yapısı kullanarak car_crashes verisindeki isminde "no" barındıran değişkenlerin
# isimlerini küçük harf yap sonuna "TAG" yazınız. barındırmayanların ismini büyüt başına başına "FOG" yazdır

df = sns.load_dataset("car_crashes")

[col.lower() + "TAG" if "no" in col else "FOG" + col.upper()  for col in df.columns]
###
#List Comprehension yapısı kullanarak verilen listedekinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

k = ["speeding", "alcohol"]
new_col = [col for col in df.columns if col not in k ]
new_df = df[new_col]
new_df.info()
new_df.head()

### pandas numpy seaborn

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
#Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
df["sex"].value_counts()

# Her bir sutuna ait unique değerlerin sayısını bulunuz.
df.columns.unique().value_counts()

#who değişkeninin unique değerlerini bulunuz.
df["who"].unique()

# survived ve age değişkenlerinin unique değerlerinin sayısını bulunuz.
df[["survived", "age"]].nunique()

# embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")

# embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"] == "C"]

# embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df["embarked"] != "S"]

#Yaşı30dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df[df["age"] < 30 ]

#Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
df[(df["fare"] > 30) & (df["age"] > 70)]

#Her bir değişkendeki boş değerlerin toplamını bulunuz.
df.isnull().sum()

# who değişkenini dataframe’den çıkarınız.
df.drop("who", axis=1, inplace=True)

# deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()

# age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df["age"].fillna(df["age"].median(), inplace=True)
df.isnull().sum()

# survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})

#30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri setinde age_line adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)

df['age_line'] = df['age'].apply(lambda x: 1 if x < 30  else 0)
df.head()
# Seaborn kütüphanesi içerisinden tips veri setini tanımlayınız.
df = sns.load_dataset("tips")
df.head(79)
# Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df["time"].value_counts()
df.groupby("time").agg({"total_bill": ["min","max","mean"]})

# Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df.groupby(["day","time"]).agg({"total_bill": ["min","max","mean"]})

# Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day('e göre toplamını, min, max ve ortalamasını bulunuz.'
df.groupby("day").agg({"total_bill": ["min","max","mean"],"tip": ["min","max","mean"]})

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                           "tip":  ["sum","min","max","mean"]})

#not:  üstteki gruplamada df[True ve false değerlere göre seçti].güne göre grupladı tip ve.. nin min max .. toplulaştırdık.

df.head()
# size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
df.loc[(df["size"] < 3) & (df["total_bill"] >10 ) , "total_bill"].mean()

#total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
df["sum_total_bill_tip"] = df["total_bill"] + df["tip"]
df.head()

# total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
n_df = df.sort_values("sum_total_bill_tip", ascending=False)[:30]
n_df = n_df.reset_index()
n_df.head(31)




