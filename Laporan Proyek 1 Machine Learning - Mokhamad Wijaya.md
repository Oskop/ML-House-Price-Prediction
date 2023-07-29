# Laporan Proyek 1 Machine Learning - Mokhamad Wijaya

## Domain Proyek

Rumah adalah sebuah kebutuhan yang terbilang pokok untuk sebuah keluarga. Rumah merupakan sebuah tempat bernaung manusia dari segala kondisi cuaca dan tempat beristirahat yang relatif paling nyaman. Dikala seseorang sudah menginjak usia dewasa dan memantapkan hati untuk berkeluarga, maka sebuah rumah merupakan pertimbangan dalam menentukan manajemen finansialnya. Tak terbatas pada kebutuhan primer manusia, dalam bisnis perumahan, diperlukan beberapa analisis dan pertimbangan dalam menentukan harga rumah. Apabila harga rumah yang ditentukan tidak berdasarkan analisis atau hanya menggunakan perkiraan kasar tanpa data yang valid, maka sebuah usah yang bergerak di bidang perumahan dapat menjual harga rumah yang terlalu murah dari harga pasar sehingga menyebabkan profit yang sedikit, atau saat dijual dengan harga yang terlalu tinggi maka para calon pelanggan akan enggan untuk membeli.

Dengan berkembangannya teknologi, khususnya dibagian *machine learning* dapat memprediksi harga rumah berdasarkan dataset yang tersedia. Dengan prediksi berdasarkan data, maka hasil yang dikeluarkan oleh model *machine learning* diharapkan dapat membantu dalam menentukan harga rumah yang sesuai. Tujuan utama dari pengembangan model ini adalah untuk membangun sistem yang dapat memprediksi harga rumah dengan relatif akurat.

Harga rumah memainkan peran penting dalam membentuk perekonomian. Renovasi dan konstruksi rumah merangsang ekonomi dengan meningkatkan tingkat penjualan rumah, lapangan kerja, dan pengeluaran. Proses prediksi harga tradisional yang membosankan berdasarkan perbandingan harga jual dan prediksi proses acak sulit dicapai dengan akurasi yang berharga. Dikarenakan hal tersebut, maka dibutukan model prediksi untuk menentukan harga rumah yang realistis dan relatif tepat berdasarkan data yang sudah ada.[1]

## Business Understanding

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, penulis akan mengembangkan sebuah sistem prediksi harga rumah untuk menjawab permasalahan berikut.

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh dalam menentukan harga rumah?
- Berapa harga rumah yang relevan dengan karakteristik atau fitur tertentu? 

### Goals

Untuk  menjawab pertanyaan tersebut, penulis membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Mengetahui fitur yang paling tidak berkorelasi atau berpengaruh terhadap harga sebuah rumah.
- Membuat model *machine learning* yang dapat memprediksi harga rumah dengan seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements

- Untuk mengetahui fitur mana yang paling tidak berkorelasi dalam memprediksi harga rumah yaitu dengan metode *Multivariate analysis* yang mana digunakan untuk menghitung nilai korelasi antar fitur harga rumah.
- untuk mendapat hasil yang akurat, penelitian perlu menggunakan beberapa algoritma. Berhubung data yang digunakan bersifat regresi, setiap model akan diukur dengan metrik ``mean_squared_error()``. Skala ``mean_squared_error()`` berada di rentang _n_ sampai 0 yang mana _n_ menunjukkan seberapa jauh nilai eror model dan 0 menunjukkan model sangat akurat. Model yang menunjukan nilai eror yang paling rendah adalah model yang relatif akurat dalam memprediksi harga rumah.

## Data Understanding

Data yang digunakan dalam proyek ini adalah [**Dataset Prediksi Harga Rumah**](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction). Dataset prediksi harga rumah adalah kumpulan data 545 harga rumah dengan 12 fitur yang mempengaruhinya. Fitur tersebut mencakup *area*, *bedrooms*, *bathrooms*, *stories*, *mainroad*, *guestroom*, *basement*, *hot water heating*, *Airconditioning*, *Parking*, *prefarea*, dan *furnishing status*. Kumpulan data ini dapat digunakan untuk membangun model *machine learning* untuk memprediksi harga rumah berdasarkan sejumlah fitur dari yang telah disebutkan. 

### Variabel-variabel pada Dataset Prediksi Diabetes adalah sebagai berikut:

- Price: harga dari sebuah rumah.

- Area: total luas rumah dalam satuan _feet_ persegi.

- Bedrooms: Jumlah kamar tidur di rumah.

- Bathrooms: Jumlah kamar mandi di dalam rumah.

- Stories: Jumlah cerita atau kisah yang ada pada rumah.

- Mainroad: apakah rumah terhubung ke jalan utama (Yes/No).

- Guestroom: apakah rumah memiliki ruang tamu (Yes/No).

- Basement: apakah rumah memiliki _basement_ (Yes/No).

- Hot water heating: apakah rumah memiliki sistem pemanas air (Yes/No).

- Airconditioning: apakah rumah memiliki sistem penyejuk udara atau AC (Yes/No).

- Parking: jumlah ruang/lahan parkir yang tersedia di dalam rumah.

- Prefarea: apakah rumah terletak di area pilihan (Yes/No).

- Furnishing status: status perabotan rumah (Fully Furnished, Semi-Furnished, Unfurnished).


Terdapat beberapa tahapan dalam memahami dataset tersebut, yaitu:

1. Periksa duplikasi

    Duplikasi pada dataset dapat mempengaruhi hasil prediksi harga rumah menjadi buruk. Oleh karena hal tersebut, dilakukan pengecekan duplikasi data.

    Berikut adalah hasil pengecekan duplikasi

    >>> jumlah baris duplikasi:  0

    Setelah pengecekan, tidak ditemukan duplikasi data sehingga data masih berjumlah 545.

    

2. Periksa *missing value*

    _Missing value_ pada fitur-fitur dataset dapat membuat hasil prediksi menjadi buruk.

    Seletah dilakukan pengecekan _missing value_, tidak ada fitur yang mengalami hal tersebut.
    
    karena tidak ada _missing value_, maka jumlah data tetap 545.


3. *Multivariate Analysis*

    Analisis multivariat merujuk pada kumpulan teknik statistik yang digunakan untuk menganalisis dan memahami hubungan antara beberapa fitur secara simultan. Berbeda dengan analisis univariat (yang berfokus pada satu fitur) atau analisis bivariat (yang mempelajari hubungan antara dua fitur), analisis multivariat mempertimbangkan interaksi dan ketergantungan antara tiga atau lebih fitur. Dalam kasus ini, analisis multivariate digunakan seberapa besar korelasi semua fitur terhadap fitur harga rumah agar fitur yang memiliki korelasi yang kecil dapat disingkirkan.
    

    Untuk evaluasi skor korelasi antar fitur khususnya fitur numerik, peneliti menggunakan fungsi `corr()` lalu skor yang didapat akan ditampilkan dalam diagram heatmap agar mudah dibaca.
    
    ![Screenshot_20230611_231927](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/a27338eb-6a61-4993-850a-7dbebe00e9f0)
    
    Gambar 1. Matriks korelasi pada fitur numerik

    Dikarenakan tidak semua fitur berupa numerik, penulis melakukan standardidasi pada fitur-fitur kategorial dan didapat diagram heatmap untuk semua fitur.

    ![Screenshot_20230611_231927](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/a27338eb-6a61-4993-850a-7dbebe00e9f0)
    
    Gambar 2. Matriks korelasi pada fitur numerik

    Berdasarkan gambar 2 di atas, dari hasil analisis, fitur ``hotwaterheating``, ``guestroom``, dan ``basement`` memiliki korelasi paling kecil karena skornya mendekati angka 0. Koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah. Selanjutnya fitur fitur ``hotwaterheating``, ``guestroom``, dan ``basement`` di-*drop*. Maka sekarang fitur yang tersisa adalah `price`, `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `airconditioning`, `parking`, `prefarea` dan `furnishingstatus`. 

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Berikut persiapan data yang dilakukan yaitu:

1. Encoding Fitur Kategori

    Setelah melakukan analisis, tersisa beberapa fitur kategorikal dan numerik. fitur kategorikal perlu diubah ke fitur numerik agar model dapat memperlakukan setiap kategori secara terpisah dan tidak memberikan nilai peringkat atau urutan yang tidak relevan. Dengan cara ini, kita dapat menggunakan data kategori dalam model *machine learning* atau analisis data dengan lebih efektif.

    Terdapat tipe data kategori atau bukan numerik pada fitur mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea dan furnishingstatus. Dikarenakan model hanya dapat memproses data numerik, maka perlu pengubahan bentuk data dari fitur-fitur tersebut ke dalam bentuk numerik.

    Untuk tipe data kategori dengan 2 nilai yang unik (mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea) maka akan diubah menjadi skala 0 (no) dan 1 (yes)

    Untuk tipe data kategori yang memiliki lebih dari 2 nilai unik, maka akan diurutkan berdasarkan nilai maknanya. Pada kasus fitur furnishingstatus, yang memiliki 3 nilai unik yang dapat diurutkan dari nilai kecil ke nilai besar (Unfurnished, Semi-Furnished, Fully Furnished) maka akan diwakilkan dalam bentuk numerik secara beruntun dengan nilai (0, 1, 2)

2. Train-Test-Split

    Setelah semua fitur berupa numerik, kemudian lakukan *train-test-split*. *train-test-split* adalah metode yang digunakan dalam pembelajaran mesin dan statistik untuk membagi data menjadi dua subset yang saling terpisah, yaitu data pelatihan (training data) dan data pengujian (testing data).

    Selanjutnya pilih fitur target atau fitur yang akan dijadikan prediksi yaitu fitur `diabetes` lalu tandai sebagai variabel "**y**". Fitur-fitur selain fitur target ditandai dengan variabel "**X**".

    *Sklearn* sudah menyiapakan fungsi untuk membagi data latih dan data uji yaitu ``train_test_split()``. Lalu lakukan pemisahan dengan fungsi tersebut, maka data tersebut akan terpisah dengan ditandai nama variabel *train* dan *test*. 
    
    Rasio pembagian data latih dan data uji yaitu 80:10 karena dataset memiliki sedikit data. Total data sebelumnya sebesar 545. Setelah dibagi dengan rasio 80:10, maka total data latih sebesar 436 dan data uji sebesar 109.

    Dengan membagi data menjadi subset pelatihan dan pengujian, kita dapat menghindari penilaian yang terlalu optimis dan mendapatkan perkiraan yang lebih realistis tentang seberapa baik model akan berperforma pada data yang tidak pernah dilihat sebelumnya.

3. Standarisasi

    Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.
 
    StandardScaler adalah salah satu teknik standarisasi yang umum digunakan dalam machine learning. Teknik ini berguna untuk mengubah distribusi data menjadi memiliki mean (rerata) 0 dan standar deviasi (simpangan baku) 1. Dengan menggunakan StandardScaler, setiap fitur akan diperlakukan secara independen dan diskala ulang sehingga memiliki skala yang seragam.

    fitur-fitur yang akan dilakukan standarisasi yaitu ``'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'`` agar fitur-fitur tersebut memiliki skala yang seragam.

    Jika fitur-fitur memiliki skala yang berbeda-beda, interpretasi hasil model dapat menjadi sulit. Misalnya, sulit untuk membandingkan bobot atau koefisien antar fitur jika mereka memiliki skala yang berbeda. Juga, interpretasi pengaruh setiap fitur dalam model dapat menjadi lebih rumit jika skala tidak seragam.


## Modeling

Sebelum memulai *modeling*, siapkan terlebih dahulu *dataframe* untuk menyimpan hasil evaluasi model. Evaluasi model pada proyek ini menggunakan metrik error dengan fungsi `mean_squared_error`. Dataframe terdiri dari kolom dan baris. Nama kolom terdiri dari `train_mse` dan `test_mse`. Nama baris terdiri dari nama-nama algoritma yang digunakan.

Pada tahap *modeling*, digunakan beberapa algoritma klasifikasi untuk memprediksi status diabetes pasien yaitu *K-Nearest Neighbor Regression*, *Random Forest Regression*, *Linear Regression*.

- **K-Nearest Neighbor Regression**

    K-Nearest Neighbor (K-NN) Regression adalah salah satu metode klasifikasi yang populer dalam machine learning. Pendekatan ini digunakan untuk mengklasifikasikan data baru berdasarkan kesamaan atau jaraknya terhadap data yang ada dalam dataset pelatihan. Dalam K-NN, kata "K" mengacu pada jumlah tetangga terdekat yang akan digunakan untuk menentukan kelas atau label data baru.

    Kelebihan K-NN Regression:

    1. Sederhana dan mudah dipahami: Konsep K-NN relatif sederhana dan mudah dipahami. Hal ini membuatnya cocok untuk pemula dalam machine learning.

    2. Tidak memerlukan proses pembelajaran: K-NN termasuk dalam metode pembelajaran berbasis instansi (instance-based), yang berarti tidak memerlukan proses pelatihan yang kompleks. Model K-NN secara efektif "menghafal" data pelatihan dan menggunakannya langsung saat melakukan prediksi.

    3. Toleran terhadap perubahan data: K-NN dapat dengan mudah menyesuaikan diri dengan perubahan dalam data pelatihan. Jika ada penambahan data baru, model K-NN tidak perlu dilatih ulang, melainkan hanya menambahkan data tersebut ke dataset yang ada.

    Kekurangan K-NN Regression:

    1. Komputasi yang mahal: Ketika dataset menjadi sangat besar, K-NN dapat menjadi komputasi yang mahal. Menghitung jarak antara data baru dengan setiap titik data dalam dataset pelatihan bisa memakan waktu yang cukup lama.

    2. Sensitif terhadap data yang tidak relevan: K-NN dapat menjadi sensitif terhadap data yang tidak relevan dalam dataset. Data yang tidak relevan dapat menyebabkan distorsi dalam perhitungan jarak dan menghasilkan prediksi yang tidak akurat.

    3. Memerlukan pemrosesan data yang tepat: Sebelum menggunakan K-NN, sering kali diperlukan pemrosesan data untuk menghilangkan atribut yang tidak relevan, mengisi nilai yang hilang, atau melakukan normalisasi. Pemrosesan data yang tepat dapat membantu meningkatkan kinerja K-NN.
    
    Parameter yang digunakan :
    
    - `n_neighbors`: Parameter ini menentukan jumlah tetangga terdekat yang akan digunakan dalam proses klasifikasi. Nilai yang umumnya digunakan adalah bilangan ganjil untuk menghindari situasi kesetimbangan kelas. Jumlah tetangga yang terlalu rendah dapat menghasilkan model yang sensitif terhadap noise, sementara jumlah tetangga yang terlalu tinggi dapat menghasilkan model yang terlalu umum. Untuk proyek ini, model ini menggunakan nilai 10. Artinya, KNN Classifier akan mempertimbangkan 10 tetangga terdekat dalam menghasilkan prediksi.

    Hasil dari pelatihan model K-NN Classifier pada data latih akan dievaluasi menggunakan metrik `mean_squared_error()` dan akan disimpan pada variabel `models.loc['train_mse', 'KNN']`.

- **Random Forest Regression**

    Random Forest Regression adalah sebuah algoritma yang digunakan dalam machine learning untuk melakukan klasifikasi data. Algoritma ini bekerja dengan menggabungkan beberapa pohon keputusan (decision trees) yang bekerja secara independen dan menghasilkan prediksi berdasarkan mayoritas suara atau rata-rata dari prediksi masing-masing pohon.

    Berikut adalah beberapa kelebihan dari Random Forest Regression:

    1. Akurasi yang Tinggi: Random Forest Regression sering memberikan hasil yang sangat akurat dalam klasifikasi data. Hal ini disebabkan oleh fakta bahwa algoritma ini menggabungkan prediksi dari banyak pohon keputusan, sehingga dapat mengurangi overfitting dan mengatasi bias yang ada dalam setiap pohon individu.

    2. Toleransi terhadap Fitur Irrelevan: Algoritma ini mampu menangani dataset dengan fitur yang tidak relevan atau tidak penting. Saat membangun setiap pohon keputusan, hanya sebagian kecil dari fitur yang digunakan secara acak untuk membatasi pengaruh fitur yang kurang penting terhadap prediksi.

    3. Keandalan terhadap Noise: Random Forest Regression dapat menangani dataset yang mengandung noise atau outlier. Ketika prediksi dibuat dengan memperhitungkan mayoritas suara dari pohon-pohon individu, dampak dari data yang salah atau noise dapat dikurangi.

    4. Kecepatan dan Skalabilitas: Algoritma ini dapat bekerja dengan cepat pada dataset besar dan memiliki kemampuan untuk memproses banyak fitur. Dengan mengoptimalkan paralelisme, Random Forest Regression dapat dijalankan secara efisien pada sistem dengan sumber daya yang cukup.

    Namun, Random Forest Regression juga memiliki beberapa kelemahan:

    1. Tidak Interpretatif: Hasil dari Random Forest Regression mungkin sulit diinterpretasikan. Karena algoritma ini menggabungkan prediksi dari banyak pohon keputusan, sulit untuk menguraikan bagaimana setiap fitur berkontribusi terhadap prediksi akhir.

    2. Kompleksitas Model: Random Forest Regression membangun banyak pohon keputusan, yang dapat menyebabkan kompleksitas model yang tinggi. Hal ini dapat menyebabkan waktu pelatihan yang lama, terutama untuk dataset yang besar, serta memerlukan sumber daya komputasi yang cukup tinggi.

    3. Overfitting pada Data dengan Fitur Lebih Banyak: Jika dataset memiliki jumlah fitur yang sangat besar dibandingkan dengan jumlah sampel, Random Forest Regression dapat mengalami overfitting. Dalam kasus ini, algoritma cenderung terlalu menyesuaikan diri dengan data pelatihan dan kinerjanya pada data yang tidak dikenal dapat menurun.

    4. Pengaturan Parameter yang Penting: Random Forest Regression memiliki beberapa parameter yang perlu dikonfigurasi dengan benar, seperti jumlah pohon dalam ensemble dan kedalaman maksimum setiap pohon. Pengaturan parameter yang tidak tepat dapat mempengaruhi kinerja model secara keseluruhan.

    Parameter yang digunakan :
    
    - `n_estimators`: Parameter ini menentukan jumlah pohon keputusan yang akan dibangun dalam model Random Forest. Semakin banyak pohon yang digunakan, semakin kompleks model dan waktu pelatihannya akan meningkat. Untuk proyek ini, model ini menggunakan nilai 50. Artinya, Random Forest Regression akan menggunakan 50 pohon keputusan dalam menghasilkan prediksi.
    - `max_depth`: Parameter ini menentukan kedalaman maksimum dari setiap pohon dalam model. Kedalaman yang lebih dalam dapat menghasilkan model yang lebih kompleks, tetapi juga dapat menyebabkan overfitting. Jika tidak ditentukan, pohon tidak memiliki batasan kedalaman dan akan terus membagi sampai semua node menjadi murni atau sampai jumlah sampel minimum untuk split tidak terpenuhi. Untuk proyek ini, model tidak menentukan parameter `max_depth`, untuk itu model akan menggunakan nilai 16.
    - `random_state`: mengontrol pengacakan bootstrap sampel yang digunakan saat membangun pohon-pohon keputusan (jika `bootstrap`=True) dan pengambilan sampel fitur untuk dipertimbangkan saat mencari pemisahan terbaik di setiap node (jika `max_features` < `n_features`). Untuk model ini menggunakan nilai 55.
    - `n_jobs`: jumlah tugas yang dijalankan secara paralel. Jika nilai `None` sama dengan 1 tugas dalam satu waktu, bila -1 sama dengan semua tugas dijalankan secara paralel. Untuk model ini menggunalan nilai -1.

    Hasil dari pelatihan model Random Forest Regression pada data latih akan dievaluasi menggunakan metrik `mean_squared_error()` dan akan disimpan pada variabel `models.loc['train_mse', 'RandomForest']`.

- **Linear Regression**

    Analisis regresi linier digunakan untuk memprediksi nilai suatu variabel berdasarkan nilai variabel lain. Variabel yang ingin Anda prediksi disebut variabel dependen. Variabel yang Anda gunakan untuk memprediksi nilai variabel lain disebut variabel independen. Bentuk analisis ini mengestimasi koefisien persamaan linier, yang melibatkan satu atau lebih variabel independen yang paling baik memprediksi nilai variabel dependen. Regresi linier cocok dengan garis lurus atau permukaan yang meminimalkan perbedaan antara nilai output yang diprediksi dan yang sebenarnya. Ada kalkulator regresi linier sederhana yang menggunakan metode "kuadrat terkecil" untuk menemukan garis yang paling cocok untuk satu set data berpasangan. Kemudian memperkirakan nilai X (variabel dependen) dari Y (variabel independen).

    Beberapa kelebihan dari Linear Regression sehingga membuat metode ini masih tetap digunakan adalah sebagai berikut:

    1. Kemudahan untuk digunakan: Salah satu kelebihannya adalah metode ini cukup simpel dan mudah dipahami, namun tetap menghasilkan insight yang powerful.

    2. Menentukan Kekuatan Prediktor: Analisis regresi dapat mengidentifikasi sekuat apa pengaruh yang diberikan oleh variabel prediktor (variabel independen) terhadap variabel lainnya (variabel dependen).

    3. Dapat Memprediksi Tren di Masa yang Akan Datang: Kelebihan selanjutnya dari metode ini adalah dapat digunakan untuk memprediksi nilai yang ada pada masa depan. Ini sejalan dengan fungsi dari analisis regresi yang dapat digunakan untuk peramalan dan prediksi.

    Kekurangan Linear Regression:

    Tentunya ketika memiliki kelebihan, pasti akan ada kekurangannya. Kekurangan yang paling mencolok adalah karena hasil ramalan dari analisis regresi merupakan nilai estimasi, sehingga kemungkinan untuk tidak sesuai dengan data aktual tetaplah ada. Selain itu, penentuan variabel independen dan variabel dependen yang saling berkaitan dalam hal sebab-akibat juga terbilang cukup susah, karena bisa jadi model yang tidak cukup bagus disebabkan karena kesalahan dalam memilih variabel yang digunakan untuk analisis. Misalkan data gaji pegawai tidak berkaitan dengan tempat anaknya bersekolah, sehingga jika menggunakan variabel tersebut model yang didapatkan tidak akan bagus.

    Parameter yang digunakan :
    
    Semua parameter yang digunakan pada model ini adalah default.

    Hasil dari pelatihan model Linear Regression pada data latih akan dievaluasi menggunakan metrik `mean_squeared_error()` dan akan disimpan pada variabel `models.loc['train_mse', 'LinearRegression']`.

## Evaluation

Untuk mengevaluasi model yang telah dilatih, proyek ini menggunakan metrik `error`. Metrik eror adalah ukuran yang digunakan untuk mengevaluasi sejauh mana model machine learning melakukan kesalahan prediksi pada data yang telah diberikan.

### Metrik Error

Metrik eror adalah salah satu metrik evaluasi yang digunakan dalam model machine learning untuk mengukur seberapa besar nilai error model dalam melakukan prediksi. Nilai eror yang dimaksud adalah selisih antara nilai sebenarnya dan nilai prediksi.

Untuk menghitung nilai eror, langkah pertama adalah membandingkan setiap prediksi yang dilakukan oleh model dengan nilai sebenarnya. Kemudian, hitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.

Berikut rumus matematika untuk menghitung akurasi:

![Screenshot_20230611_231927](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/a27338eb-6a61-4993-850a-7dbebe00e9f0)
    
Gambar 3. Rumus Mean Squared Error

Keterangan:

N = jumlah dataset

yi = nilai sebenarnya

y_pred = nilai prediksi

Pada proyek ini, nilai eror dari setiap model akan digunakan untuk membandingkan model mana yang terbaik. Rentang pada nilai eror yaitu dari 0 sampai n. Nilai error model yang paling mendekati 0 akan menjadi model yang terbaik.

Hasil dari evaluasi model menggunakan metrik error sebagai berikut:

![Screenshot_20230611_231927](https://github.com/fiqihalfito/ML-Diabetes/assets/112739333/a27338eb-6a61-4993-850a-7dbebe00e9f0)

Gambar 4. Garfik nilai error data latih dan data uji

Berdasarkan gambar di atas, dapat diambil beberapa kesimpulan:

1. Model KNN memiliki nilai eror yang cukup tinggi pada data uji dan memiliki nilai eror lebih rendah pada data latih. Dalam hal ini, nilai MSE model pada data latihan adalah 1,1144, sedangkan pada data uji adalah 1,572. Hal ini menunjukkan bahwa model KNN belum berhasil menghasilkan prediksi yang akurat pada data baru yang belum pernah dilihat sebelumnya.

2. Model Random Forest memiliki nilai eror yang sangat tinggi pada data uji dengan nilai 1,6958 dan nilai eror yang sangat rendah dengan nilai 0,2295, yang menunjukkan bahwa model ini mengalami overfitting. Hal ini menunjukkan bahwa model Random Forest tidak berhasil menghasilkan prediksi yang akurat pada data baru yang belum pernah dilihat sebelumnya.

3. Model Linear Regression menunjukkan nilai eror yang tinggi pada data uji (1,1696) dan data uji (1,3803). Hal ini menunjukkan bahwa model Linear Regression belum mampu memberikan prediksi yang akurat pada data baru yang belum pernah dilatih sebelumnya, namun karena selisih antara nilai error data uji dan data latihnya paling rendah, maka terdapat potensi untuk menghasilkan model yang tida overfitting.

## Kesimpulan

Dalam kesimpulan, semua model yang dievaluasi dalam tabel tersebut memiliki nilai eror yang relatif tinggi. Namun, model Random Forest memiliki nilai eror terendah pada data latihan, sekaligus nilai eror yang paling tinggi pada data uji. Pilihan model terbaik tergantung pada konteks dan kebutuhan spesifik dari masalah yang dihadapi. Sehingga model terbaik yang dipilih yaitu model *Linear Regression* karena dapat menunjukan hasil yang tidak mengandung banyak selisih nilai eror pada evaluasi data uji dan data latih.

## Referensi

[1]     Chen, X., Wei, L., & Xu, J. (2017). House Price Prediction Using LSTM. [Available](https://arxiv.org/abs/1709.08432v1)