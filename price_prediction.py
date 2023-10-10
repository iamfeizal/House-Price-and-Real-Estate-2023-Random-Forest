# %% [markdown]
# # Laporan Proyek Machine Learning - Submission 1: Predictive Analytics
# - Nama: Imam Agus Faisal
# - Email: imamagusfaisal120@gmail.com
# - Id Dicoding: imamaf

# %% [markdown]
# ## Problem Statement

# %% [markdown]
# - Fitur apa yang paling berpengaruh terhadap harga rumah?
# - Berapa harga rumah dengan karakteristik atau fitur tertentu?

# %% [markdown]
# ## Menyiapkan semua library yang dibutuhkan

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# %% [markdown]
# ## Data Wrangling

# %% [markdown]
# **Memuat setiap tabel pada dataset**

# %%
house_df=pd.read_excel("dataset.xlsx")
house_df.head()

# %% [markdown]
# ### Assessing Data
# **Melihat informasi, Memeriksa missing value, Memeriksa duplikasi, dan Memeriksa parameter statistik pada setiap tabel**

# %%
print('\n', house_df.info())
print('\nMissing value house:\n', house_df.isnull().sum())
print('\nJumlah duplikasi house:\n', house_df.duplicated().sum())
print('\n\nParameter statistik house:\n', house_df.describe())

# %%
sns.boxplot(x=house_df['Price'])

# %%
sns.boxplot(x=house_df['Beds'])

# %%
sns.boxplot(x=house_df['Bath'])

# %%
sns.boxplot(x=house_df['Sq.Ft'])

# %% [markdown]
# **Rangkuman Hasil Analisis Tahap Assesing Data pada Dataset:**
# *   Terdapat missing value pada kolom Place dan Website
# *   Terdapat beberapa kolom yang mempunyai Outliers
# *   Tidak terdapat adanya duplikasi
# *   Tidak terdapat innacurate value
# 
# 

# %% [markdown]
# ### Cleaning Data
# ##### Missing Value
# Karena dataset yang dimiliki cukup banyak, maka metode yang akan kita gunakan untuk menangani missing value adalah dengan menghapus/drop baris yang terdapat missing value tersebut.

# %%
house_df.loc[(house_df['Place'].isna())]

# %%
house_df.loc[(house_df['Website'].isna())]

# %%
house_df.dropna(axis=0, inplace=True)

# %% [markdown]
# Mengecek kembali dataset yang sudah dibersihkan

# %%
house_df.info()

# %%
print('\nMissing value house:\n', house_df.isnull().sum())

# %% [markdown]
# ##### Outliers

# %% [markdown]
# Hilangkan outliers yang terdapat pada kolom Price, Beds, Bath, dan Sq.Ft

# %%
Q1 = house_df['Price'].quantile(0.25)
Q3 = house_df['Price'].quantile(0.75)
IQR = Q3-Q1
lower = house_df['Price'] < Q1 - (1.5*IQR)
higher = house_df['Price'] > Q3 + (1.5*IQR)

house_df.drop(house_df[lower].index, inplace=True)
house_df.drop(house_df[higher].index, inplace=True)

# %%
sns.boxplot(x=house_df['Price'])

# %%
Q1 = house_df['Beds'].quantile(0.25)
Q3 = house_df['Beds'].quantile(0.75)
IQR = Q3-Q1
lower = house_df['Beds'] < Q1 - (1.5*IQR)
higher = house_df['Beds'] > Q3 + (1.5*IQR)

house_df.drop(house_df[lower].index, inplace=True)
house_df.drop(house_df[higher].index, inplace=True)

# %%
sns.boxplot(x=house_df['Beds'])

# %%
Q1 = house_df['Bath'].quantile(0.25)
Q3 = house_df['Bath'].quantile(0.75)
IQR = Q3-Q1
lower = house_df['Bath'] < Q1 - (1.5*IQR)
higher = house_df['Bath'] > Q3 + (1.5*IQR)

house_df.drop(house_df[lower].index, inplace=True)
house_df.drop(house_df[higher].index, inplace=True)

# %%
sns.boxplot(x=house_df['Bath'])

# %%
Q1 = house_df['Sq.Ft'].quantile(0.25)
Q3 = house_df['Sq.Ft'].quantile(0.75)
IQR = Q3-Q1
lower = house_df['Sq.Ft'] < Q1 - (1.5*IQR)
higher = house_df['Sq.Ft'] > Q3 + (1.5*IQR)

house_df.drop(house_df[lower].index, inplace=True)
house_df.drop(house_df[higher].index, inplace=True)

# %%
sns.boxplot(x=house_df['Sq.Ft'])

# %%
house_df.info()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# ### Univariate Analysis

# %%
numerical_features = ['Price', 'Beds', 'Bath', 'Sq.Ft']
categorical_features = ['Address', 'Description', 'Place', 'Website']

# %% [markdown]
# ##### Categorical Features

# %% [markdown]
# Fitur Address

# %%
feature = categorical_features[0]
count = house_df[feature].value_counts()
percent = 100*house_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

# %% [markdown]
# Fitur Description

# %%
feature = categorical_features[1]
count = house_df[feature].value_counts()
percent = 100*house_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

# %% [markdown]
# Fitur Place

# %%
feature = categorical_features[2]
count = house_df[feature].value_counts()
percent = 100*house_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

# %% [markdown]
# Fitur Website

# %%
feature = categorical_features[3]
count = house_df[feature].value_counts()
percent = 100*house_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

# %% [markdown]
# **Dengan mengamati beberapa barchart di atas, kita memperoleh beberapa informasi, antara lain:**
# *   Fitur 'Address' memiliki data dengan persebarang paling luas dan bisa dikatakan sebagai unique data.
# *   Categorical Features pada dataset ini memiliki persebaran yang cukup luas dan cukup sulit untuk membuat kategori yang cukup berpengaruh untuk fitur 'Price'.

# %% [markdown]
# ##### Numerical Features

# %%
house_df.hist(bins=40, figsize=(15,10))
plt.show()

# %% [markdown]
# **Dengan mengamati histogram di atas, khususnya variabel price yang merupakan fitur target, kita memperoleh beberapa informasi, antara lain:**
# *   Peningkatan 'price' sebanding dengan penurunan jumlah sample. Hal ini dapat kita lihat jelas dari histogram 'price' yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
# *   Distribusi 'price' miring ke kanan (right-skewed)

# %% [markdown]
# ### Multivariate Analysis

# %% [markdown]
# ##### Categorical Features

# %%
house_df.info()

# %%
cat_features = house_df.select_dtypes(include='object').columns.to_list()
 
for col in cat_features:
  sns.catplot(x=col, y="Price", hue=col, kind="bar", dodge=False, height = 4, aspect = 4,  data=house_df, palette="Set1", legend=False)
  plt.title("Rata-rata 'price' Relatif terhadap - {}".format(col))

# %% [markdown]
# **Dengan mengamati rata-rata price relatif terhadap fitur kategori di atas, kita memperoleh insight sebagai berikut:**
# *   Pada fitur ‘Address’, secara umum, memiliki persebaran yang sangat luas dan bahkan bisa dikatakan sebagai data unique dan memiliki pengaruh rendah terhadap 'price'.
# *   Pada fitur ‘Description’, secara umum, hampir setiap rumah memiliki deskripsinya masing-masing dan memiliki pengaruh yang rendah terhadap 'price'.
# *   Pada fitur ‘Place’, secara umum, persebaran pada fitur Place lebih sedikit dibanding dengan 'Address' dan 'Description'. Namun, fitur 'Place' masih memiliki pengaruh yang rendah terhadap 'price'.
# *   Pada fitur ‘Website’, secara umum, fitur 'website' tidak memiliki pengaruh yang signifikan terhadap 'price' dan fitur 'website' lebih menunjukkan rata-rata 'price' yang cenderung mirip.
# >**Kesimpulan akhir, fitur kategori memiliki pengaruh yang rendah terhadap price.**

# %% [markdown]
# Karena Categorical Features memiliki pengaruh yang rendah terhadap price, maka kita akan hapus atau drop kolom yang memiliki Categorical Features tersebut. 

# %%
house_df.drop(categorical_features, inplace=True, axis=1)

# %%
house_df.info()

# %% [markdown]
# ##### Numerical Features

# %%
sns.pairplot(house_df, diag_kind = 'kde')

# %%
plt.figure(figsize=(10, 8))
correlation_matrix = house_df[numerical_features].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

# %% [markdown]
# **Dengan mengamati pairplot dan correlation matrix terhadap price, kita memperoleh insight sebagai berikut:**
# *   Fitur ‘Sq.Ft’, memiliki hubungan korelasi positif terhadap fitur 'price'.
# *   Fitur ‘Sq.Ft’, memiliki korelasi yang paling besar dengan fitur 'price'.
# *   Fitur ‘Bath’ dan 'Beds', secara umum, memiliki korelasi sedang terhadap fitur 'price'.
# *   Fitur ‘Beds’, memiliki hubungan korelasi paling kecil terhadap fitur 'price'.
# >**Kesimpulan akhir, tidak ada fitur yang akan didrop karena masih memiliki hubungan korelasi sedang.**

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Encoding Fitur Kategori
# Pada tahap ini, kita akan melewatkan proses encoding categorical features karena categorical features pada dataset tidak terlalu berpengaruh terhadap 'Price' dan sudah kita drop, sehingga pada dataset hanya terdapat numerical features.

# %% [markdown]
# ### Reduksi Dimensi dengan PCA

# %%
sns.pairplot(house_df[['Beds','Bath','Sq.Ft']], plot_kws={"s": 3});

# %%
plt.figure(figsize=(10, 8))
correlation_matrix = house_df[['Beds','Bath','Sq.Ft']].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk 'Sq.Ft', 'Beds', 'Bath'", size=20)

# %% [markdown]
# Berdasarkan hasil dari Pairplot dan Correlation Matrix diatas, kita juga akan melewatkan proses reduksi dimensi menggunakan PCA karena tidak ada fitur yang memiliki korelasi sangat tinggi.

# %% [markdown]
# ### Train-Test-Split
# Kita akan membagi dengan proporsi pembagian sebesar 80:20 seperti pada referensi yang kita gunakan.

# %%
X = house_df.drop(["Price"],axis =1)
y = house_df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size= 0.8, random_state = 123)

# %%
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

# %% [markdown]
# ### Standarisasi
# Mengubah nilai fitur agar mendekati distribusi normal agar algoritma machine learning memilki performa yang baik.

# %%
numerical_features = ['Beds', 'Bath', 'Sq.Ft']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

# %%
X_train[numerical_features].describe().round(4)

# %% [markdown]
# ## Model Development

# %%
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['RF1', 'RF2', 'RF3'])

# %%
RF1 = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=15, n_jobs=-1)
RF1.fit(X_train, y_train)
 
models.loc['train_mse','RF1'] = mean_squared_error(y_pred=RF1.predict(X_train), y_true=y_train)          

# %%
RF2 = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=40, n_jobs=-1)
RF2.fit(X_train, y_train)
 
models.loc['train_mse','RF2'] = mean_squared_error(y_pred=RF2.predict(X_train), y_true=y_train)          

# %%
RF3 = RandomForestRegressor(n_estimators=2000, max_depth=1000, random_state=1500, n_jobs=-1)
RF3.fit(X_train, y_train)
 
models.loc['train_mse','RF3'] = mean_squared_error(y_pred=RF3.predict(X_train), y_true=y_train)          

# %% [markdown]
# ## Evaluasi Model

# %% [markdown]
# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1

# %%
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# %% [markdown]
# Membuat Dataframe berisi nilai mse train dan test

# %%
mse = pd.DataFrame(columns=['train', 'test'], index=['RF1', 'RF2', 'RF3'])
model_dict = {'RF1': RF1, 'RF2': RF2, 'RF3': RF3}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e5 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e5

mse

# %%
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

# %%
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['hasilprediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)



