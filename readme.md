# Laporan Proyek Machine Learning - Imam Agus Faisal

## Domain Proyek
### Latar Belakang
Pembuatan model machine learning untuk memprediksi harga rumah dan real estate sangat penting dalam industri properti dan real estate. Model semacam ini digunakan untuk memberikan perkiraan atau prediksi harga properti, yang dapat sangat membantu para pembeli, penjual, investor, dan pemangku kepentingan lain dalam membuat keputusan yang lebih informasi.
Berikut adalah beberapa alasan utama mengapa model machine learning untuk memprediksi harga rumah dan real estate penting:
- **Volume Data yang Besar**: Industri real estate menghasilkan volume data yang besar, termasuk informasi tentang harga, lokasi, ukuran, kondisi fisik, fasilitas, dan banyak faktor lain yang memengaruhi harga properti. Model machine learning dapat memproses dan menganalisis data ini dengan lebih efisien daripada metode tradisional.
- **Kepentingan Bisnis**: Baik penjual maupun pembeli real estate memiliki kepentingan bisnis dalam mengetahui harga yang adil dan akurat untuk suatu properti. Model machine learning dapat membantu dalam menentukan harga jual yang tepat bagi penjual dan mengevaluasi apakah suatu properti berada dalam kisaran harga yang sesuai untuk pembeli.
- **Investasi dan Portofolio**: Para investor real estate seringkali memiliki portofolio yang terdiri dari beberapa properti. Model machine learning dapat membantu dalam mengelola portofolio ini dengan memberikan perkiraan kinerja dan nilai properti dalam jangka waktu tertentu.
- **Efisiensi Proses Penilaian**: Proses penilaian properti dapat menjadi tugas yang rumit dan memakan waktu. Model machine learning dapat memberikan perkiraan cepat dan objektif tentang harga properti, yang dapat digunakan sebagai dasar awal untuk penilaian lebih lanjut.
- **Faktor-Faktor Penentu Harga yang Kompleks**: Ada banyak faktor yang memengaruhi harga properti, termasuk lokasi, tipe properti, kondisi pasar, tren ekonomi, dan faktor-faktor lingkungan. Model machine learning dapat mengidentifikasi pola dan hubungan kompleks antara faktor-faktor ini.
- **Tren Pasar dan Prediksi Masa Depan**: Model machine learning dapat digunakan untuk menganalisis tren pasar sebelumnya dan memprediksi pergerakan harga di masa depan. Hal ini berguna bagi para pemangku kepentingan yang ingin membuat keputusan investasi jangka panjang.
- **Kemajuan Teknologi**: Teknologi komputasi dan kecerdasan buatan semakin canggih. Dengan adanya data yang lebih baik dan algoritma yang lebih efisien, model machine learning dapat memberikan hasil yang lebih akurat dan cepat.

**Rubrik/Kriteria Tambahan**:

### Mengapa Masalah Tersebut Harus Diselesaikan?
Masalah prediksi harga rumah dan real estate perlu diselesaikan karena memiliki dampak signifikan pada berbagai aspek kehidupan masyarakat dan ekonomi. Berikut beberapa alasan mengapa masalah ini penting:
- **Pengaruh pada Keputusan Keuangan Individu**: Harga rumah dan real estate adalah investasi besar bagi individu. Keputusan pembelian atau penjualan properti seringkali mengikuti perubahan harga. Kemampuan untuk memprediksi harga dengan tepat dapat membantu individu membuat keputusan finansial yang lebih cerdas dan menghindari kerugian.
- **Pengaruh pada Stabilitas Keuangan**: Perubahan harga properti dapat memengaruhi stabilitas keuangan individu dan keluarga. Ketika harga properti mengalami fluktuasi yang ekstrem, dapat terjadi masalah seperti kerugian dalam transaksi jual beli, kebangkrutan, atau kesulitan membayar hipotek.
- **Pengaruh pada Investasi dan Ekonomi Makro**: Industri real estate adalah komponen penting dalam perekonomian suatu negara. Perubahan harga properti dapat berdampak pada kebijakan ekonomi, kesejahteraan masyarakat, dan stabilitas pasar keuangan. Kesalahan dalam memprediksi tren harga properti dapat memicu permasalahan ekonomi yang lebih besar.

### Penelitian yang Dijadikan Referensi Yaitu:
- Evaluasi prediksi dilakukan dengan melihat hasil error pada RMSE setiap model. Dari ketiga model tersebut (Metode Regresi Linier, Random Forest Regression dan *Gradient Boosted Trees Regression* Method), diperoleh hasil **nilai error terkecil pada algoritma Random Forest sebesar 0.440** [^1].
- Dengan algoritma linear regression untuk memprediksi harga rumah dapat memberikan hasil keakuratan prediksi harga rumah dengan baik. Adapun **hasil akurasi terbaik dengan menggunakan data set untuk training sebesar 80% dan menggunakan 20% untuk data testing** [^2].
- Penggunaan hasil evaluasi model yang menggunakan PCA memiliki tingkat error yang lebih kecil dan nilainya lebih konsisten dibandingkan dengan hasil evaluasi tanpa PCA. Waktu pelatihan menggunakan model PCA memiliki waktu yang lebih cepat dibandingkan hanya menggunakan random forest. **Penggunaan PCA dan Random Forest memiliki hasil yang lebih optimal dibandingkan dengan yang hanya menggunakan Random Forest** [^3].

## Business Understanding
Dalam bagian ini dijelaskan mengenai manfaat dari pembuatan prediksi harga rumah dan real estate  dan pentingnya memahami gejolak harga pasar yang berpengaruh terhapat dukungan keputusan bisnis. Penjelasan mengenai aspek tersebut yaitu:
- **Dukungan Keputusan Bisnis**: Para pemangku kepentingan dalam industri real estate, termasuk pengembang, investor, dan agen properti, memerlukan perkiraan harga yang akurat untuk membuat keputusan bisnis yang tepat. Prediksi harga yang akurat dapat membantu menghindari investasi yang berisiko tinggi dan meningkatkan efisiensi operasional.
- **Menghindari Gejolak Pasar**: Membuat prediksi yang tepat tentang harga properti dapat membantu menghindari gejolak pasar dan kemungkinan gelembung properti yang merugikan. Dengan informasi yang lebih baik, pihak-pihak terkait dapat mengambil tindakan preventif jika diperlukan.

### Problem Statements
Berdasarkan latar belakang yang telah diuraikan sebelumnya, dikembangkanlah sebuah sistem prediksi harga rumah untuk menjawab permasalahan berikut:
- **Fitur apa yang paling berpengaruh terhadap harga rumah?**<br>
Fitur yang ada pada rumah dan real estate sangat beragam. Maka dari itu dibutuhkanlah identifikasi mengenai fitur apa yang paling berpengaruh terhadap harga tersebut. Misalnya jumlah kamar tidur dan luas rumah.
- **Berapa harga rumah dengan karakteristik atau fitur tertentu?**<br>
Dengan beragam karakteristik yang dipunyai setiap rumah, tentu penentuan harga rumah akan sangat sulit bagi pemilik agar harga tersebut sesuai dengan harga pasar saat ini. Penentuan harga yang sesuai pasar juga dapat digunakan untuk menghindari gejolak pasar.

### Goals
Untuk  menjawab pertanyaan tersebut, dibuatlah *predictive modelling* dengan tujuan atau *goals* sebagai berikut:
- **Mengetahui fitur yang paling berkorelasi atau berpengaruh terhadap harga rumah**<br>
Tujuan dari pernyataan masalah pertama adalah dengan diketahui fitur yang paling berkorelasi atau berpengaruh terhadap harga rumah, diharapkan dapat memudahkan pemilik untuk dijadikan patokan dalam penentuan harga rumah agar sesuai harga pasar.
- **Membuat model *machine learning* untuk memprediksi harga rumah seakurat mungkin berdasarkan fitur yang berpengaruh**<br>
Tujuan dari pernyataan masalah kedua adalah mengembangkan model prediksi harga rumah dan real estate menggunakan *machine learning* untuk memprediksi harga rumah berdasarkan fitur atau karakteristik yang berpengaruh agar memudahkan pemilik dalam menentukan harga sesuai pasar dan menghindari gejolak pasar.

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
- *Address* : Alamat lengkap dimana rumah tersebut didirikan.
- *Price* : Harga rumah tersebut.
- *Description* : Deskripsi yang menggambarkan fitur atau karakteristik rumah.
- *Place* : Wilayah rumah tersebut berdiri (Kota rumah tersebut didirikan).
- *Beds* : Jumlah tempat tidur yang ada didalam rumah tersebut.
- *Bath* : Jumlah kamar mandi yang ada dirumah tersebut (Dalam rumah tentu tidak semua kamar tidur mempunyai kamar mandi, hal tersebut digambarkan dengan nilai 0.5).
- *Sq.Ft* : Luas keseluruhan rumah tersebut dengan satuan Square Foot.
- *Website* : Alamat *website* tempat rumah tersebut ditawarkan/dijual.

**Rubrik/Kriteria Tambahan**:
### Exploratory Data Analysis:
Dalam bagian ini akan dijelaksan mengenai cara menangani missing value dan outliers yang terdapat dalam data dan analisis univariate dan multivariate, yaitu:
- **Univariate analysis** bertujuan untuk memahami dan menganalisis satu variabel atau fitur pada suatu waktu. Ini digunakan untuk menggambarkan distribusi, karakteristik, dan statistik dasar dari satu variabel tertentu.
- **Multivariate analysis** bertujuan untuk memahami hubungan kompleks antara dua atau lebih variabel dalam dataset. Ini digunakan untuk mengeksplorasi bagaimana variabel-variabel tersebut berinteraksi satu sama lain.
#### Menangani *Missing Value* dan *Outliers*
##### Missing Value
Missing Value pada kolom *'Place'*:
|      |                Address |   Price |   Description | Place | Beds | Bath | Sq.Ft |                          Website |
|------|-----------------------:|--------:|--------------:|------:|-----:|-----:|------:|---------------------------------:|
|  122 | 1066 Creekside Blvd SW |  580900 |  CA AB T2X5K6 |   NaN |    1 |  1.5 |   964 |             Maxwell Canyon Creek |
|  641 |  62 Royston Terrace NW |  849900 | CA AB T3L 0J2 |   NaN |    3 |  2.5 |  2467 |                       Cir Realty |
| 1154 |          3250 84 St SE | 2400000 | CA AB T2B 3C1 |   NaN |    6 |  2.0 |  2147 |                       RE/MAX Key |
| 1174 |    8535 19 Ave SE #424 |  455000 | CA AB T2A 7W8 |   NaN |    2 |  1.5 |  1193 |                       Exp Realty |
| 1245 |     99 Taralake Way NE |  672000 |  CA AB T3J0A7 |   NaN |    5 |  3.5 |  1749 |               One Percent Realty |
| 1324 |      148 Savanna Dr NE |  850000 |  CA AB T3J2H5 |   NaN |    4 |  3.0 |  2315 |                  Maxwell Central |
| 1519 |            9110 34 Ave | 2299000 | CA AB T1X 0L5 |   NaN |    6 |  4.5 |  2034 |                      Real Broker |
| 1630 |  71 Lynx Meadows Dr NW | 2000000 | CA AB T3L 3L9 |   NaN |    6 |  4.5 |  3688 |           Greater Property Group |
| 1846 |    8535 19 Ave SE #421 |  505000 | CA AB T2A 7W8 |   NaN |    3 |  2.5 |  1428 |                       Exp Realty |
| 1867 |         4520 84 Ave NE | 1000000 | CA AB T3J 4C4 |   NaN |    5 |  4.0 |  2753 |          Century 21 Bravo Realty |
| 2019 |     99 Royston Rise NW |  794900 | CA AB T3L 0J2 |   NaN |    3 |  2.5 |  2037 |                       Cir Realty |
| 2361 | 1161 Creekside Blvd SW |  667500 |  CA AB T2X5K5 |   NaN |    3 |  2.5 |   945 |             Maxwell Canyon Creek |
| 2629 |    8535 19 Ave SE #417 |  488900 | CA AB T2A 7W8 |   NaN |    4 |  3.5 |  1365 |                       Exp Realty |
| 3033 |      30 Forzani Way NW | 2388000 | CA AB T3Z 1L5 |   NaN |    5 |  3.5 |  2554 |                       Cir Realty |
| 3311 |    902 Bluerock Way SW |  702500 | CA AB T2Y 0S5 |   NaN |    3 |  2.5 |  2096 |                             Bode |
| 3356 |        4111 162 Ave SW | 8000000 | CA AB T2Y 0N7 |   NaN |    5 |  4.5 |  9031 | Diamond Realty & Associates Ltd. |

Missing value pada kolom *'Website'*:
|     |              Address |  Price |   Description |  Place | Beds | Bath | Sq.Ft | Website |
|-----|---------------------:|-------:|--------------:|-------:|-----:|-----:|------:|--------:|
| 160 | 341 Walcrest View SE | 820000 | CA AB T2X 4V9 | Walden |    5 |  3.5 |  2235 |     NaN |

**Kesimpulan:**<br>
Terdapat nilai null atau *missing value* pada kolom Place dan Website. Karena kita memiliki dataset yang cukup banyak, maka kita bisa menghilangkannya dengan menghapus baris yang mempunyai *missing value* tersebut. 

##### Outliers
| ![](/assets/images/box_price.png) <center><b>Gambar 1</b> - Boxplot Price</center> | ![](/assets/images/box_beds.png) <center><b>Gambar 2</b> - Boxplot Beds</center> |
|---|--:|
| ![](/assets/images/box_bath.png) <center><b>Gambar 3</b> - Boxplot Bath</center> | ![](/assets/images/box_sq.png) <center><b>Gambar 4</b> - Boxplot Sq.Ft</center> |

**Kesimpulan:**<br>
Terdapat beberapa *outliers* pada data numerical yang dapat diketahui menggunakan boxplot. *Outliers* tersebut dapat kita hilangkan dengan menghapus atau drop baris yang mempunyai nilai pencilan tersebut.

#### Univariate Analysis
##### Categorical Features<br>
| ![](/assets/images/uni_address.png) <center><b>Gambar 5</b> - Univariate Address</center>| ![](/assets/images/uni_desc.png) <center><b>Gambar 6</b> - Univariate Description</center> |
|---|--:|
| ![](/assets/images/uni_place.png) <center><b>Gambar 7</b> - Univariate Place</center> | ![](/assets/images/uni_web.png) <center><b>Gambar 8</b> - Univariate Website</center> |

**Kesimpulan:**<br>
Dengan mengamati barchart, kita memperoleh beberapa informasi, antara lain:
- Fitur 'Address' memiliki data dengan persebarang paling luas dan bisa dikatakan sebagai unique data.
- Categorical Features pada dataset ini memiliki persebaran yang cukup luas dan cukup sulit untuk membuat kategori yang cukup berpengaruh untuk fitur 'Price'.
##### Numerical Features<br>
![](/assets/images/uni_numeric.png)
<center><b>Gambar 9</b> - Univariate Numeric Features</center>

**Kesimpulan:**<br>
Dengan mengamati histogram pada Univariate, khususnya variabel price yang merupakan fitur target, kita memperoleh beberapa informasi, antara lain:
- Peningkatan 'price' sebanding dengan penurunan jumlah sample. Hal ini dapat kita lihat jelas dari histogram 'price' yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
- Distribusi 'price' miring ke kanan (right-skewed)
#### Multivariate Analysis
##### Categorical Features<br>
| ![](/assets/images/multi_address.png) <center><b>Gambar 10</b> - Multivariate Address</center> |
|---|
| ![](/assets/images/multi_desc.png) <center><b>Gambar 11</b> - Multivariate Decription</center> |
| ![](/assets/images/multi_place.png) <center><b>Gambar 12</b> - Multivariate Place</center> |
| ![](/assets/images/multi_web.png) <center><b>Gambar 13</b> - Multivariate Website</center> |

**Kesimpulan:**<br>
Dengan mengamati rata-rata price relatif terhadap fitur kategori, kita memperoleh insight sebagai berikut:
- Pada fitur ‘Address’, secara umum, memiliki persebaran yang sangat luas dan bahkan bisa dikatakan sebagai data unique dan memiliki pengaruh rendah terhadap 'price'.
- Pada fitur ‘Description’, secara umum, hampir setiap rumah memiliki deskripsinya masing-masing dan memiliki pengaruh yang rendah terhadap 'price'.
- Pada fitur ‘Place’, secara umum, persebaran pada fitur Place lebih sedikit dibanding dengan 'Address' dan 'Description'. Namun, fitur 'Place' masih memiliki pengaruh yang rendah terhadap 'price'.
- Pada fitur ‘Website’, secara umum, fitur 'website' tidak memiliki pengaruh yang signifikan terhadap 'price' dan fitur 'website' lebih menunjukkan rata-rata 'price' yang cenderung mirip.
- **Kesimpulan akhir, fitur kategori memiliki pengaruh yang rendah terhadap price.**
##### Numerical Features<br>
![](/assets/images/multi_numeric.png)
<center><b>Gambar 14</b> - Univariate Numeric Features</center>
<br><br>

![](/assets/images/corr_numeric.png)
<center><b>Gambar 15</b> - Correlation Matrix</center>

**Kesimpulan:**<br>
Dengan mengamati pairplot dan correlation matrix terhadap price pada Gambar 14 dan Gambar 15, kita memperoleh insight sebagai berikut:
- Fitur ‘Sq.Ft’, memiliki hubungan korelasi positif terhadap fitur 'price'.
- Fitur ‘Sq.Ft’, memiliki korelasi yang paling besar dengan fitur 'price'.
- Fitur ‘Bath’ dan 'Beds', secara umum, memiliki korelasi sedang terhadap fitur 'price'.
- Fitur ‘Beds’, memiliki hubungan korelasi paling kecil terhadap fitur 'price'.
- **Kesimpulan akhir, tidak ada fitur yang akan didrop karena masih memiliki hubungan korelasi sedang.**

## Data Preparation
Pada tahap data preparation, kita akan mengubah data categorical (**jika ada**) menjadi bentuk yang dapat digunakan dalam algoritma machine learning dan mereduksi dimensi dengan PCA (**jika data mempunyai korelasi yang sangat tinggi**).
- **Tujuan Encoding Categorical Features (Encoding Fitur Katgori)**:<br>
Ketika bekerja dengan data kategorikal (seperti jenis kelamin, kategori produk, atau kode pos), perlu mengubahnya menjadi bentuk yang dapat digunakan dalam algoritma machine learning yang umumnya membutuhkan data numerik. Encoding kategorikal features adalah cara untuk mengubah data ini menjadi representasi numerik yang sesuai.
- **Tujuan Reduksi Dimensi menggunakan PCA (Principal Component Analysis)**:<br>
PCA digunakan untuk mengurangi dimensi dalam dataset yang memiliki banyak fitur (variabel) dan untuk mengidentifikasi pola utama dalam data. Ini berguna untuk mengurangi kompleksitas data, mempercepat algoritma machine learning, dan menghilangkan multicollinearity (korelasi tinggi antara variabel).
**Rubrik/Kriteria Tambahan**:
### Encoding Fitur Kategori
Pada tahap ini, kita akan melewatkan proses encoding categorical features karena categorical features pada dataset tidak terlalu berpengaruh terhadap 'Price' dan sudah kita drop, sehingga pada dataset hanya terdapat numerical features.
### Reduksi Dimensi dengan PCA
![](/assets/images/corr_numeric.png)
<center><b>Gambar 16</b> - Correlation Matrix</center>

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
<center><b>Gambar 17</b> - Bar Chart Evaluation</center>

Berdasarkan hasil evaluasi yang dapat dilihat pada Gambar 2, model terbaik ternyata pada model RF1. Hal tersebut berarti model dengan hyperparameter **n_estimator dan max_depth dengan nilai tinggi bukan berarti akan selalu memberi model terbaik**. Hal tersebut **dapat terjadi karena adanya Overvitting** pada model.


Hasil Output:

|      | y_true | hasilprediksi_RF1 | hasilprediksi_RF2 | hasilprediksi_RF3 |
|------|-------:|------------------:|------------------:|------------------:|
| 1915 | 694000 |          252847.2 |          192374.0 |          190106.1 |


## Conclusion
Hasil prediksi model menunjukkan angka prediksi harga yang cukup jauh. Hal tersebut bisa saja terjadi karena dataset yang diberikan mempunyai sebaran data yang cukup besar dan tidak memiliki korelasi yang sangat berpengaruh terhadap 'Price'.

![](/assets/images/corr_numeric.png)
<center><b>Gambar 18</b> - Correlation Matrix Numerik</center>

Hal tersebut dapat kita amati dari hasil nilai korelasi pada Gambar 3 yang menunjukkan bahwa hanya 1 Fitur yang memiliki nilai korelasi cukup besar, yaitu fitur 'Sq.Ft' dan fitur tersebut hanya memiliki nilai < 0.85.
Selain itu, fitur yang tersisa hanya meiliki nilai korelasi yang sedang atau bahkan rendah.

> [!IMPORTANT]
> Maka dari itu, wajar saja jika hasil prediksi harga rumah mempunyai harga yang cukup jauh, karena bukan dari modelnya melainkan dari fitur didalam dataset yang memiliki persebaran cukup luas dan nilai korelasi yang cenderung rendah


## Daftar Referensi

[^1]: [Analisis Perbandingan Metode Regresi Linier, Random Forest Regression dan Gradient Boosted Trees Regression Method untuk Prediksi Harga Rumah](https://journal.isas.or.id/index.php/JACOST/article/download/491/202)

[^2]: [Prediksi Harga Rumah Menggunakan Web Scrapping dan Machine Learning dengan Algoritma Linear Regression](http://jurnal.mdp.ac.id/index.php/jatisi/article/download/701/219)

[^3]: [Optimasi Metode Random Forest Menggunakan Principal Component Analysis Untuk Memprediksi Harga Rumah](http://etheses.uin-malang.ac.id/50422/1/210605220005.pdf)
