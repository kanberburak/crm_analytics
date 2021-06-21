import pandas as pd
# Veri seti online_retail_II.xlsx okutuldu.
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_ = pd.read_excel("/Users/pc/Desktop/vbo/pythonProject/online_retail_II.xlsx",
                    sheet_name = "Year 2010-2011")

# Veriseti kopyalandı.
df = df_.copy()

# NA, Shape, Head, Tail gözlemlenmek için check_df fonksiyonu kullanıldı.
def check_df(dataframe, head=5):
    """
    It gives information about shape of the dataframe, type of its variables, first 5 observations, last 5 observations,
    total number of missing observations and quartiles.

    Parameters:
    ----------
        dataframe: dataframe
            The dataframe whose properties will be defined
        head: int, optional
            Refers to the number of observations that will be displayed from the beginning and end

    Returns
    -------
        The return statement does not exist.
    Examples:
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        check_df(df, 10)
    or
        check_df(df)
    Notes:
    ------
        The number of observations to be viewed from the end is the same as the number of observations to be observed
        from the beginning.
    """
    print("########## SHAPE ##########")
    print(dataframe.shape)
    print("########## TYPES ##########")
    print(dataframe.dtypes)
    print("########## HEAD ##########")
    print(dataframe.head(head))
    print("########## TAIL ##########")
    print(dataframe.tail(head))
    print("########## NA ##########")
    print(dataframe.isnull().sum())
    print("########## QUANTILES ##########")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

# Betimsel istatistikler için describe metodu kullanıldı.
df.describe().T

# count     mean     std       min      25%      50%      75%  \
# Quantity    541910.00     9.55  218.08 -80995.00     1.00     3.00    10.00
# Price       541910.00     4.61   96.76 -11062.06     1.25     2.08     4.13
# Customer ID 406830.00 15287.68 1713.60  12346.00 13953.00 15152.00 16791.00

# 541910 tane gözlem sayısı, 8 tane değişken mevcuttur.
# 4 tane kategorik, 3 tane sayısal değişken ve tarih değişkeni var.
# 4 tane object, 1 tane int, 2 tane float ve datetime değişkeni var.
# customerID değişkeninde 135080 tane NA gözlem var.
# Description değişkeninde 1454 tane NA gözlem var.

"""
Veriseti hikayesi:
InvoiceNo – Fatura Numarası 
Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.
StockCode – Ürün kodu
Her bir ürün için eşsiz numara.
Description – Ürün ismi
InvoiceDate – Fatura tarihi
UnitPrice – Fatura fiyatı (Sterlin)
CustomerID – Eşsiz müşteri numarası
Country – Ülke ismi
Quantity – Ürün adedi
Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
"""

df.isnull().sum()
df.dropna(inplace = True)
df.shape
# Eksik değerler çıkartılınca gözlem sayısı 541910'dan 406830'a düştü.

df["StockCode"].nunique()
# 3684 adet eşsiz ürün vardır.

df["Description"].value_counts()
# Örnek: WHITE HANGING HEART T-LIGHT HOLDER ürününden 2070 tane vardır.

df.groupby("StockCode").agg("count")["Quantity"].sort_values(ascending = False).head(5)
# En çok sipariş edilen 5 ürünü çoktan aza doğru groupby ve aggregation kullanarak sıraladık.

df = df[~df["Invoice"].str.contains("C", na=False)]
# İade ve iptal edilen ürünleri string contains ile "C"'leri yakalayarak ~ile çıkarttık.

df["TotalPrice"] = df["Quantity"] * df["Price"]
# Fatura başına elde edilen toplam kazancı ifade eden 'TotalPrice' değişkeni oluşturuldu.

#############
RFM Metriklerini Hesaplayalım
#############

df["InvoiceDate"].max()
import datetime as dt
today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ["recency", "frequency", "monetary"]

rfm = rfm[rfm["monetary"] > 0]

rfm.head()

############################
RFM Skorlarının Hesaplanması
############################
# Recency
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels = [5,4,3,2,1])

# Frequency
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels = [1,2,3,4,5])

# Monetary
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels = [1,2,3,4,5])

rfm["RFM_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
# + rfm["monetary_score"].astype(str) dahil etmedik.
rfm.head()

###########################################
RFM skorlarının segment olarak tanımlanması
###########################################
# RFM isimlendirme

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex = True)
rfm.head()
# seg_map yardımı ile skorlar segmentlere çevrildi.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
# oh be :D valla yoruldum saat 3:14 AM hocam okuyorsanız selam neyse devam

##############################
##     AKSİYON ZAMANI       ##
##############################
rfm[rfm['segment'] == 'cant_loose'].describe().T
# !!!!!!! ACİL DESTEK ATALIM
rfm[rfm['segment'] == 'need_attention'].describe().T
# !!!!!!! DURUM CİDDİ :D
rfm[rfm['segment'] == 'loyal_customers'].describe().T
# !!!!!!! LOYALLER İLGİ BEKLER.
"""
Öncelikle 'Champions' segmentinde olan müşterilerimiz velinimetimiz ancak ilgilenmemiz gereken
daha önemli kişiler var. Champ'ler zaten bize ilgi duyuyorlar. <3
Amacımız diğer segmentlerden Champion yetiştirmek olmalı;

######################
Cant_Lose
Need_Attention         WHAT'S THE PROBLEM GARDAS ?
Loyal_Customer
######################

######################
Championlarımızdan sonra en çok bize sterlin kazandıran müşterilerimiz, 
LOYAL CUSTOMERS! kendilerini seviyoruz, 
+ Recency Mean: 33.61 yani en son 1 ay önce satış yapmışız.
+ Loyal_customers müşterilerimizin ilgilendiklerini ürünleri analiz edip, hiç ilgilenmeyeceği ürünleri
satma girişimini pazarlama departmanına tavsiye ederim. Çünkü loyal_customers'lar bizim ürünlerimizden düzenli
alıyorsa bu kişiler bizim gerçekten sadık müşterilerimiz ise zaten aldıkları ürünleri almaya devam edecekler.
Champion yapmalıyız. Bizi düzenli ziyaret eden bu müşterilerimiz, geldiğinde onlara çok alakasız şeyleri satabilmek 
için önceden biraz subliminal mesaj içeren reklam tasarlayabiliriz, örnek olarak pandemi döneminde dışarıdan gözlemlediğim
kadarıyla "KAHVE İÇİN TERMOS-BARDAK" satışlarında ciddi bir artış var ama sebebini bilmiyorum. Sürekli bende alacak oluyorum ama almadım hala.
Pazarlama bölümünün işine daha fazla karışmadan devam ediyorum :) dipnot: Loyal_customer'ların hangi ürünleri aldığını 
nasıl bulabileceğimi, eğer atölye olursa sormak ve tartışmak için buraya not düşüyorum, bu bizim işimiz.

CANT_LOSE! grubu ortalama 132.97 gündür yok! ATTENTION ATTENTION! 63 kişi nerede ? 8.38 frequency ortalaması ile
Champion'larımızdan sonra en çok alışveriş yapan segmentimiz kendileri, 2796.16 ortalama monetary(birim bırakmışlar)
loyal_customers 2864, 133 gündür olmayan cant_lose segmentini canlandırmamız lazım, 100 liralık alışverişe 1 aylık
netflix üyeliği verebiliriz, dizi-film izledikçe bizi hatırlarlar. Kaybetme lüksümüz yok sizi CANT_LOSE.

Need_Attention! SUPPORT! SUPPORT! 187 kişi olan bu segmentimizin recency mean'i 52, ortalama 52 gündür yoklar. 
frequency 2.33 düşük olabilir ama bıraktıkları 897 birim iyi. Alışveriş sıklığı az olmasına rağmen bırakılan birim 
miktar fazla ? O zaman Need_attetion'da bulunan 187 müşterimiz biraz temkinli alışveriş yapabiliyor olabilir. 
Örnek: ben :D, elektronik bir cihaz alcağımı düşünüyorum diyelim, iyice araştırıp, bütçem doğrultusunda Fiyat/Perf.
ararım. Marka değeri çok pahalı olan bir ürünü fazla fiyat vermek yerine daha ucuzunu ve işimi görecek olanı alırım.
2 tane ürün alsam ve ortalama 50 gün sonra tekrar alışveriş yapmam için ne gerekir ? Kendime soruyorum evet, Burak?
Mail, mesaj, tatil hediyesi , bonuslar, çok ilgimi çekmez! İlgimi çekecek olan şey marka değeri pahalı olup, benim
zamanında alamadığım ama hep takip ettiğim ürünler olabilir evet kesinlikle! Need_Attention segmenti için pazarlama
departmanımıza acilen, marka değeri toplum tarafından onaylanmış, hatta bir statü göstergesi gibi görülen bazı ürünlerin
fiyatlarını internet üzerinden alınacak fiyatın altına çekilmesini ve eğer bu indirim zarara sürüklüyorsa, yanında
alınan ürüne bir süs ilave edilmesini, örneğin: Kılıf ve bu kılıftan zararın 2 katını karşılamalarını tavsiye ederim.
Tabi yine bu iş pazarlamanın işi biz elimizden geleni yaptık. 
Bu yorumlamayı bu şekilde yaparsam sanırım beni işten kovarlar, lol
# Daha ciddi bir üslupla yazmak isterdim ama şuan hem eğlendim, hem öğrendim, hem de mutluyum.
######################
Teşekkürler VBO AİLESİ
######################
"""
loyal_customers = pd.DataFrame()
loyal_customers["loyal_customers_id"] = rfm[rfm["segment"] == "loyal_customers"].index
loyal_customers.to_csv("loyal_customers_csv")
