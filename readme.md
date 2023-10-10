# Laporan Proyek Machine Learning - Imam Agus Faisal

## Domain Proyek

Rumah merupakan kebutuhan salah satu kebutuhan pokok manusia. Kebutuhan akan tempat tinggal menjadi salah satu yang harus dipersiapkan manusia terutama untuk rencana jangka panjang. Dalam mempersiapkan rencana tersebut tentunya banyak karakteristik/fitur yang mempengaruhi termasuk harga, lokasi, luas, dan sebagainya. Maka dari itu, dibuatlah Model *Machine Learning* - Predictive Learning agar dapat memudahkan manusia dalam merencanakan kebutuhan tempat tinggal dengan mempertimbangkan karakteristik/fitur yang mempengaruhinya.

**Rubrik/Kriteria Tambahan**:
- Masalah tersebut harus diselesaikan agar dapat memudahkan manusia dalam merencanakan kebutuhan tempat tinggal dengan mempertimbangkan karakteristik/fitur yang mempengaruhinya.
- Evaluasi prediksi dilakukan dengan melihat hasil error pada RMSE setiap model. Dari ketiga model tersebut (Metode Regresi Linier, Random Forest Regression dan *Gradient Boosted Trees Regression* Method), diperoleh hasil **nilai error terkecil pada algoritma Random Forest sebesar 0.440** [^1].
- Dengan algoritma linear regression untuk memprediksi harga rumah dapat memberikan hasil keakuratan prediksi harga rumah dengan baik. Adapun **hasil akurasi terbaik dengan menggunakan data set untuk training sebesar 80% dan menggunakan 20% untuk data testing** [^2].
- Penggunaan hasil evaluasi model yang menggunakan PCA memiliki tingkat error yang lebih kecil dan nilainya lebih konsisten dibandingkan dengan hasil evaluasi tanpa PCA. Waktu pelatihan menggunakan model PCA memiliki waktu yang lebih cepat dibandingkan hanya menggunakan random forest. **Penggunaan PCA dan Random Forest memiliki hasil yang lebih optimal dibandingkan dengan yang hanya menggunakan Random Forest** [^3].

## Business Understanding

Saya adalah seorang praktisi *machine learning* ingin membuat sebuah model *machine learning* yang dapat membantu manusia untuk menentukan harga rumah berdasarkan karakteristik/fitur yang melekat pada rumah tersebut.

### Problem Statements

Berdasarkan latar belakang yang telah diuraikan sebelumnya, saya akan mengembangkan sebuah sistem prediksi harga rumah untuk menjawab permasalahan berikut:
- Fitur apa yang paling berpengaruh terhadap harga rumah?
- Berapa harga rumah dengan karakteristik atau fitur tertentu?

### Goals

Untuk  menjawab pertanyaan tersebut, saya akan membuat predictive modelling dengan tujuan atau *goals* sebagai berikut:
- Mengetahui fitur yang paling berkorelasi atau berpengaruh terhadap harga rumah
- Membuat model *machine learning* untuk memprediksi harga rumah seakurat mungkin berdasarkan fitur yang berpengaruh

**Rubrik/Kriteria Tambahan**:
  ### Solution statements
  - Menggunakan Model Algoritma *Random Forest*, karena model terbaik untuk membuat prediksi harga rumah adalah model *Random Forest* karena memiliki nilai error paling kecil daripada model regresi linear dan *Gradient Boosted Trees Regression* [^1].
  - Menggunakan rasio perbandingan dataset untuk training dan testing sebesar 80% dan 20%, karena hasil akurasi terbaik menggunakan rasio pembagian dataset tersebut [^2].
  - Menggunakan PCA untuk mengoptimalkan *Random Forest* **(Jika pada dataset terdapat fitur yang memiliki korelasi sangat tinggi)** [^3].

## Data Understanding
Pada model kali ini, saya menggunakan dataset dari Kaggle yaitu dataset *Housing Price & Real Estate* - 2023 dari [reenapinto](https://www.kaggle.com/reenapinto).
Dataset ini berisi 3360 Baris dengan 1 *Header* dan 8 Kolom, yaitu: *Address*, *Price*, *Description*, *Place*, *Beds*, *Bath*, *Sq.Ft*, dan *Website*.

Sumber Dataset: [Housing Price & Real Estate - 2023](https://www.kaggle.com/datasets/reenapinto/housing-price-and-real-estate-2023).

### Variabel-variabel pada *Housing Price & Real Estate* - 2023 dataset adalah sebagai berikut:
- *Address* : merupakan alamat rumah.
- *Price* : merupakan harga rumah.
- *Description* : merupakan deskripsi rumah yang akan dijual.
- *Place* : merupakan lokasi rumah tersebut.
- *Beds* : merupakan jumlah tempat tidur yang ada didalam rumah tersebut.
- *Bath* : merupakan jumlah kamar mandi yang ada dirumah tersebut.
- *Sq.Ft* : merupakan luas keseluruhan rumah tersebut.
- *Website* : merupakan *website* tempat rumah tersebut ditawarkan/dijual.

**Rubrik/Kriteria Tambahan**:
### Exploratory Data Analysis:
#### Menangani *Missing Value* dan *Outliers*
- Terdapat nilai null atau *missing value* pada kolom Place dan Website. Karena kita memiliki dataset yang cukup banyak, maka kita bisa menghilangkannya dengan menghapus baris yang mempunyai *missing value* tersebut.
- Terdapat beberapa *outliers* pada data numerical yang dapat diketahui menggunakan boxplot. *Outliers* tersebut dapat kita hilangkan dengan menghapus atau drop baris yang mempunyai nilai pencilan tersebut.
#### Univariate Analysis
- Categorical Features<br>
  Dengan mengamati barchart, kita memperoleh beberapa informasi, antara lain:
  - Fitur 'Address' memiliki data dengan persebarang paling luas dan bisa dikatakan sebagai unique data.
  - Categorical Features pada dataset ini memiliki persebaran yang cukup luas dan cukup sulit untuk membuat kategori yang cukup berpengaruh untuk fitur 'Price'.
- Numerical Features<br>
  Dengan mengamati histogram pada Univariate, khususnya variabel price yang merupakan fitur target, kita memperoleh beberapa informasi, antara lain:
  - Peningkatan 'price' sebanding dengan penurunan jumlah sample. Hal ini dapat kita lihat jelas dari histogram 'price' yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
  - Distribusi 'price' miring ke kanan (right-skewed)
#### Multivariate Analysis
- Categorical Features<br>
  Dengan mengamati rata-rata price relatif terhadap fitur kategori, kita memperoleh insight sebagai berikut:
  - Pada fitur ‘Address’, secara umum, memiliki persebaran yang sangat luas dan bahkan bisa dikatakan sebagai data unique dan memiliki pengaruh rendah terhadap 'price'.
  - Pada fitur ‘Description’, secara umum, hampir setiap rumah memiliki deskripsinya masing-masing dan memiliki pengaruh yang rendah terhadap 'price'.
  - Pada fitur ‘Place’, secara umum, persebaran pada fitur Place lebih sedikit dibanding dengan 'Address' dan 'Description'. Namun, fitur 'Place' masih memiliki pengaruh yang rendah terhadap 'price'.
  - Pada fitur ‘Website’, secara umum, fitur 'website' tidak memiliki pengaruh yang signifikan terhadap 'price' dan fitur 'website' lebih menunjukkan rata-rata 'price' yang cenderung mirip.
  - **Kesimpulan akhir, fitur kategori memiliki pengaruh yang rendah terhadap price.**
- Numerical Features<br>
  Dengan mengamati pairplot dan correlation matrix terhadap price, kita memperoleh insight sebagai berikut:
  - Fitur ‘Sq.Ft’, memiliki hubungan korelasi positif terhadap fitur 'price'.
  - Fitur ‘Sq.Ft’, memiliki korelasi yang paling besar dengan fitur 'price'.
  - Fitur ‘Bath’ dan 'Beds', secara umum, memiliki korelasi sedang terhadap fitur 'price'.
  - Fitur ‘Beds’, memiliki hubungan korelasi paling kecil terhadap fitur 'price'.
  - **Kesimpulan akhir, tidak ada fitur yang akan didrop karena masih memiliki hubungan korelasi sedang.**

## Data Preparation
**Rubrik/Kriteria Tambahan**:
### Encoding Fitur Kategori
Pada tahap ini, kita akan melewatkan proses encoding categorical features karena categorical features pada dataset tidak terlalu berpengaruh terhadap 'Price' dan sudah kita drop, sehingga pada dataset hanya terdapat numerical features.
### Reduksi Dimensi dengan PCA
![](/assets/images/corr_numeric.png)
<center><b>Gambar 1</b> - Correlation Matrix</center>

Berdasarkan hasil dari *Correlation Matrix* yang dapat dilihat pada Gambar 1, kita juga akan melewatkan proses reduksi dimensi menggunakan PCA karena tidak ada fitur yang memiliki korelasi sangat tinggi.

### Train-Test-Split
Membagi dataset dengan proporsi 80:20 karena hasil akurasi terbaik menggunakan rasio pembagian dataset tersebut [^2].
### Standarisasi
Mengubah nilai fitur agar mendekati distribusi normal agar algoritma machine learning memilki performa yang baik.

  |       |      Beds |      Bath |     Sq.Ft |
  |-------|----------:|----------:|----------:|
  | count | 2452.0000 | 2452.0000 | 2452.0000 |
  |  mean |   -0.0000 |   -0.0000 |   -0.0000 |
  |   std |    1.0002 |    1.0002 |    1.0002 |
  |   min |   -1.5719 |   -1.4941 |   -1.7607 |
  |   25% |   -0.7801 |   -0.3799 |   -0.8078 |
  |   50% |    0.0116 |    0.1772 |   -0.2351 |
  |   75% |    0.8034 |    0.7343 |    0.7283 |
  |   max |    3.1786 |    2.4057 |    3.0204 |

## Modeling
Pada tahap ini, kita lansung menggunakan Algoritma *Random Forest*.
Hal tersebut karena Algoritma *Random Forest* memiliki nilai error paling kecil daripada model regresi linear dan *Gradient Boosted Trees Regression* [^1]

**Rubrik/Kriteria Tambahan (Opsional)**: <br>
Karena kita menggunakan 1 model algoritma, maka kita akan melakukan *hyperparameter tuning*.
- Model *Random Forest* 1 (RF1), dengan *n_estimators=10, max_depth=5, random_state=15, n_jobs=-1*
- Model *Random Forest* 2 (RF2), dengan *n_estimators=100, max_depth=50, random_state=40, n_jobs=-1*
- Model *Random Forest* 3 (RF3), dengan *n_estimators=2000, max_depth=1000, random_state=1500, n_jobs=-1*

Pada tahap ini, kita mencari tuning yang paling optimal berdasarkan data dan melakukan training model berulang kali agar mendapatkan tuning yang maksimal.
- Disini kita **mengubah** *n_estimator*(untuk jumlah *tree* di *forest*), mengubah max_depth(untuk panjang pohon), dan mengubah *random_state*(untuk mengontrol *random* *number* *generator*).
- *Hyperparameter* yang **tidak diubah** adalah *n_jobs*, disini kita tidak mengubahnya agar pekerjaan perhitungan model dilakukan secara paralel dengan memaksimalkan *thread* pada komputer.

## Evaluation
Metrik yang digunakan untuk mengevaluasi model adalah MSE atau *Mean Squared Error*.

**Rubrik/Kriteria Tambahan (Opsional)**:<br>
*Mean Squared Error* digunakan untuk menghitung selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Berikut adalah persamaan MSE:

$$MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-y\_pred_i)^{2}$$

Sebelum menghitung nilai MSE, kita perlu melakukan proses scaling pada data uji agar skala antara data latih dan data uji sama dan bisa dilakukan evaluasi.

Hasil Evaluasi terhadap ketiga model:


|     |         train |           test |
|-----|--------------:|---------------:|
| RF1 | 210767.598803 |  2182114.71799 |
| RF2 |  53843.059563 | 2666325.929435 |
| RF3 |  52874.416766 | 2685907.592525 |

Perbandingan dengan Bar Chart:

![](/assets/images/eval_bar.png)
<center><b>Gambar 2</b> - Bar Chart Evaluation</center>

Berdasarkan hasil evaluasi yang dapat dilihat pada Gambar 2, model terbaik ternyata pada model RF1. Hal tersebut berarti model dengan hyperparameter **n_estimator dan max_depth dengan nilai tinggi bukan berarti akan selalu memberi model terbaik**. Hal tersebut **dapat terjadi karena adanya Overvitting** pada model.


Hasil Output:

|      | y_true | hasilprediksi_RF1 | hasilprediksi_RF2 | hasilprediksi_RF3 |
|------|-------:|------------------:|------------------:|------------------:|
| 1915 | 694000 |          252847.2 |          192374.0 |          190106.1 |


## Conclusion
Hasil prediksi model menunjukkan angka prediksi harga yang cukup jauh. Hal tersebut bisa saja terjadi karena dataset yang diberikan mempunyai sebaran data yang cukup besar dan tidak memiliki korelasi yang sangat berpengaruh terhadap 'Price'.

![](/assets/images/corr_numeric.png)
<center><b>Gambar 3</b> - Correlation Matrix Numerik</center>

Hal tersebut dapat kita amati dari hasil nilai korelasi pada Gambar 3 yang menunjukkan bahwa hanya 1 Fitur yang memiliki nilai korelasi cukup besar, yaitu fitur 'Sq.Ft' dan fitur tersebut hanya memiliki nilai < 0.85.
Selain itu, fitur yang tersisa hanya meiliki nilai korelasi yang sedang atau bahkan rendah.

> [!IMPORTANT]
> Maka dari itu, wajar saja jika hasil prediksi harga rumah mempunyai harga yang cukup jauh, karena bukan dari modelnya melainkan dari fitur didalam dataset yang memiliki persebaran cukup luas dan nilai korelasi yang cenderung rendah


## Daftar Referensi

[^1]: [Analisis Perbandingan Metode Regresi Linier, Random Forest Regression dan Gradient Boosted Trees Regression Method untuk Prediksi Harga Rumah](https://journal.isas.or.id/index.php/JACOST/article/download/491/202)

[^2]: [Prediksi Harga Rumah Menggunakan Web Scrapping dan Machine Learning dengan Algoritma Linear Regression](http://jurnal.mdp.ac.id/index.php/jatisi/article/download/701/219)

[^3]: [Optimasi Metode Random Forest Menggunakan Principal Component Analysis Untuk Memprediksi Harga Rumah](http://etheses.uin-malang.ac.id/50422/1/210605220005.pdf)

