#!pip install lifetimes
#pip install sqlalchemy
#!pip install mysql-connector-python-rf

import mysql.connector
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import numpy as np
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', None)
#pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

#########################
# Verinin Veri Tabanından Okunması
#########################

# credentials.
creds = {'user': 'synan_dsmlbc_group_2_admin',
         'passwd': 'iamthedatascientist*****!',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'synan_dsmlbc_group_2'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
conn = create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)

retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)

retail_mysql_df.info()
retail_mysql_df.shape
retail_mysql_df.head()
type(retail_mysql_df)
# Verinin df olarak kopyalanası
df = retail_mysql_df.copy()

df.describe().T
# United Kingdom Seçilmesi df içinde.
df=df[df["Country"]=="United Kingdom"]
#########################
# Veri Ön İsleme
#########################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

df.head()

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (cltv_df'de analiz gününe göre, burada kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                        'Invoice': lambda num: num.nunique(),
                                        'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["monetary"] > 0]

# BG-NBD modeli için recency ve T'nin haftalık cinsten ifade edilmesi
# BG-NBD (Beta Geometric Negative Binomial Distribution) model
# BG-NBD Model for Customer Base Analysis
# BG-NBD (Beta Geometrik Negatif Binom Dağılımı) modeli
# Müşteri Tabanı Analizi için BG-NBD Modeli

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# Frequency'nin 1'den büyük olması gerekiyor.
# Retention customer olması için en az 1'den büyük olmalı.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

##############################################################
# 2. BG-NBD Modelinin Kurulması
# Beta Geometric Negative Binomial Distribution
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 1 haftalık beklenen transaction
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])
cltv_df.head()
# 1 aylık beklenen transaction

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_df.head()

cltv_df.sort_values("expected_purc_1_month", ascending=False).head(20)
# nihai cltv değil, sadece satın-almaları gözlemledik.
##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.head()

###########
# GOREV 1
###########

# 6 aylık CLTV prediction

cltv_6 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,    # 6 ay olarak aldım.
                                   freq="W",  # T'nin frekans bilgisi
                                   discount_rate=0.01)

cltv_6 = cltv_6.reset_index()

cltv_6.columns = cltv_6.columns.str.replace('clv', '6_aylik_CLTV')

cltv_final = cltv_df.merge(cltv_6, on="Customer ID", how="left")

cltv_6.head()

cltv_final.sort_values(by="6_aylik_CLTV", ascending=False).head(10)

# Min - Median - Max'ı inceleyelim
cltv_final.loc[:,["Customer ID","recency","T","frequency","monetary","6_aylik_CLTV"]].describe([0.5]).T

# Top 10
cltv_final.loc[:,["Customer ID","recency","T","frequency","monetary","6_aylik_CLTV"]].sort_values(by="6_aylik_CLTV", ascending=False).head(10)


# 6 Aylık CLTV YORUM:
# İlk Top 10 incelendiğimizde, Recency ve T ortalama 52 civarında iken, 2. sıradaki müşterimizin Tenure'ı 14.57
# Tenure'ı düşük ve diğerlerinden daha az frekenas ile işlem yapmış olsa da işlem başına Monetary yüksektir.
# Bundan dolayı CLTV monetary değeri baskın olarak 2. sırada yer almıştır.
# Top_10 listemizde baskın olarak öncelikle Frekans, Monetory , Recency sırasıyla etkili olduğunu söyleyebiliriz.

"""
# GOREV 2
 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
 Fark var mı? Varsa sizce neden olabilir?
 Farklı zaman periyotlarından oluşan CLTV analizi
"""

# 1 aylık CLTV'yi var olan modele ekleyelim

cltv_1 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_1 = cltv_1.reset_index()

cltv_1.head()

cltv_1.columns = cltv_1.columns.str.replace('clv', '1_aylik_CLTV')

cltv_1.head()

cltv_final= cltv_final.merge(cltv_1, on="Customer ID", how="left")

cltv_final.head()

# 12 aylık CLTV'yi var olan modele ekleyelim

cltv_12= ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_12 = cltv_12.reset_index()

cltv_12.head()

cltv_12.columns = cltv_12.columns.str.replace('clv', '12_aylik_CLTV')

cltv_12.head()

cltv_final= cltv_final.merge(cltv_12, on="Customer ID", how="left")

cltv_final.head()

# 1 aylik CLTV TOP_10
cltv_final.loc[:,["Customer ID","recency","T","frequency","monetary","1_aylik_CLTV"]].sort_values(by="1_aylik_CLTV", ascending=False).head(10)

# 12 aylik CLTV TOP_10
cltv_final.loc[:,["Customer ID","recency","T","frequency","monetary","12_aylik_CLTV"]].sort_values(by="12_aylik_CLTV", ascending=False).head(10)

#### YORUM #####
# TOP 7 CLTV_1 ile CLTV_12 için sıralama birebir aynıdır. Diğer sıralama da oldukça benzerdir.
# 1. sıradaki müşterinin CLTV_1 Aylık: 14.884 , CLTV_12 Aylık: 163.591 dir. Linear bir ilişki olmadığı açıktır.
# CLTV_1 de 10. sırada yeni müşterinin monetory scorundan dolayı 1 AYLIK cltv iyidir. Ama 12 aylıkta TOP 10 da bu müşteri yoktur.
# 2. Sıradaki müşteri diğerlerine göre Tenure ve Frekans düşüktür ve yenidir. Ama yüksek monetory ile CLTV score yüksektir.
"""
Top 10'da CLTV_1 ve CLTV_12 için ilk 7 kişi aynıdır. 
Diğer sıralamalar benzer kişilerden oluşmaktadır.
Customer_ID 18102.0 olan müşterimizi incelediğimizde;
CLTV_1 Aylik: 14884 birim
Aralarında doğrusal bir ilişki linear'lik görmediğini gözlemliyoruz.
CLTV_12 Aylik: 163591 birim
CLTV_1 için 10. sırada yer alan müşterimizin monetary skoru 1 aylık olmak üzere gayet iyi gözükmektedir.
CLTV_12' de ise bu müşterimiz yoktur.
Tenure ve frekans düşüklüğü ile göze çarpan 2. sıradaki müşterimiz yenidir. Yüksek monetary ile CLTV skoru yüksektir.
"""

###########
# GOREV 3
###########

# CLTV'nin Standartlaştırılması
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["6_aylik_CLTV"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["6_aylik_CLTV"]])
cltv_final.head()

# 6 Aylık CLTV'ye Göre Segmentlerin Oluşturulması
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final[["6_aylik_CLTV",
            "segment",
            "monetary",
            "frequency",
            "recency","T"]].groupby('segment').agg(['mean']).reset_index()
"""
A segmentini diğerlerinden ayıran en baskın özelliği Frekans ve sonrasında Monetary'nin yüksek olmasıdır.
A ve B segmentlerinin recency değeri yakın olmasına rağmen, Frekans ve Monetary olarak ayrışmaktadırlar.
B segmenti için ek satış ve çapraz satış uygulanabilir. Frekans ve Monetary için etkisi gözlemlenecektir.
A segmentimizin yüksek kar getirdiği için Müşteri Sadakatı programı uygulayabiliriz.
D segmenti Tenure yüksek, Recency düşüktür. Genel bazda tek seferlik alışveriş için gelen müşterilerimizdir.
D segmenti için daha çok kampanya düzenleyebiliriz.
"""

###########
# GOREV 4
###########

# Düzenleme:
cltv_final = cltv_final.drop(["6_aylik_CLTV","1_aylik_CLTV","12_aylik_CLTV"],axis=1)
cltv_final.head()

# SQL ile db yazdırma
cltv_final.to_sql(name='Burak_Kanber', con=conn, if_exists='replace', index=False)

# Kontrol Edilmesi
pd.read_sql_query("show tables", conn)
