# Chapter 1 Summary: The Machine Learning Landscape

## Pengenalan Machine Learning
Bab ini dimulai dengan membantah mitos umum tentang Machine Learning (ML), yang sering digambarkan sebagai robot seperti pelayan yang andal atau Terminator yang mematikan. Namun, ML sudah ada di sini dan telah digunakan selama puluhan tahun dalam aplikasi khusus seperti Pengenalan Karakter Optik (OCR). Aplikasi ML pertama yang populer adalah filter spam pada tahun 1990-an, yang belajar dengan sangat baik sehingga pengguna jarang perlu menandai spam lagi. Hari ini, ML menggerakkan ratusan produk dan fitur, mulai dari rekomendasi hingga pencarian suara.

Bab ini menjelaskan apa itu ML: ilmu (dan seni) memprogram komputer untuk belajar dari data. Definisi meliputi:
- Arthur Samuel (1959): Bidang studi yang memberi komputer kemampuan untuk belajar tanpa diprogram secara eksplisit.
- Tom Mitchell (1997): Sebuah program belajar dari pengalaman E terkait tugas T dan ukuran kinerja P jika kinerjanya pada T meningkat seiring dengan E.

Contoh: Filter spam menggunakan data pelatihan (E: email spam/ham yang diberi label) untuk meningkatkan penandaan spam (T) yang diukur berdasarkan akurasi (P). Contoh yang digunakan untuk belajar disebut kumpulan data pelatihan, dengan setiap contoh menjadi contoh pelatihan atau sampel. Mengunduh Wikipedia hanya menambah data tetapi tidak meningkatkan tugas, jadi itu bukan ML.

Jika sudah familiar dengan dasar-dasar, lanjutkan ke Bab 2; jika tidak, jawab pertanyaan di akhir bab. Bab ini adalah gambaran umum tingkat tinggi tanpa banyak kode, memperkenalkan konsep dasar dan istilah teknis yang harus diketahui setiap ilmuwan data.

## Mengapa Menggunakan Machine Learning?
Pemrograman tradisional untuk filter spam melibatkan analisis pola (misalnya, kata-kata seperti “4U,” “kartu kredit,” “gratis,” “luar biasa” dalam baris subjek, nama pengirim, dan isi email) serta penulisan algoritma deteksi untuk masing-masing pola, pengujian, dan iterasi hingga hasilnya memadai (Gambar 1-1: Pendekatan tradisional). Hal ini menghasilkan daftar panjang aturan kompleks yang sulit untuk dipelihara.  
<img width="978" height="489" alt="image" src="https://github.com/user-attachments/assets/c7286d8e-fac2-4486-a0b0-c82d9fc2e44f" />
Sebaliknya, filter spam berbasis ML secara otomatis mempelajari prediktor yang baik dengan mendeteksi pola kata yang tidak biasa sering muncul dalam contoh spam dibandingkan dengan ham (Gambar 1-2: Pendekatan Machine Learning). Program ini lebih singkat, lebih mudah dipelihara, dan kemungkinan lebih akurat. Ia beradaptasi dengan perubahan, seperti spammer yang beralih ke “For U,” dengan mendeteksi pola baru dalam spam yang dilaporkan pengguna dan menandainya secara otomatis tanpa intervensi (Gambar 1-3: Pembelajaran Mesin dapat membantu manusia belajar).
<img width="984" height="500" alt="image" src="https://github.com/user-attachments/assets/86ddfcd2-d79d-496f-94b0-b31d71d6964a" />
<img width="985" height="383" alt="image" src="https://github.com/user-attachments/assets/46094add-718a-4927-b52b-ae9c55e86259" />

ML unggul dalam menangani masalah kompleks yang tidak memiliki algoritma yang diketahui, seperti pengenalan suara (menetapkan aturan secara kaku untuk membedakan “satu” dan “dua” tidak dapat diterapkan secara luas pada berbagai aksen, kebisingan, bahasa, dan jutaan penutur—ML belajar dari rekaman contoh).  

ML juga membantu manusia belajar melalui penambangan data: Menganalisis model yang telah dilatih dapat mengungkap korelasi yang tidak terduga atau tren baru (Gambar 1-4: Machine Learning dapat membantu manusia belajar).
<img width="986" height="511" alt="image" src="https://github.com/user-attachments/assets/5b6445a3-aa22-431f-bed2-3fa3449510ad" />

Untuk merangkum, ML sangat cocok untuk:
- Masalah yang memerlukan banyak penyesuaian halus atau daftar aturan yang panjang (satu algoritma ML seringkali menyederhanakan kode dan berkinerja lebih baik).
- Masalah kompleks di mana pendekatan tradisional tidak menghasilkan solusi yang baik (teknik ML dapat menemukan solusi).
- Lingkungan yang fluktuatif (sistem ML dapat beradaptasi dengan data baru).
- Mendapatkan wawasan tentang masalah kompleks dan jumlah data yang besar (menemukan pola tersembunyi).

## Contoh Aplikasi
Contoh konkret meliputi:
- Menganalisis gambar produk di jalur produksi untuk mengklasifikasikannya secara otomatis: Klasifikasi gambar menggunakan jaringan saraf konvolusional (CNN; lihat Bab 14).
- Mendeteksi tumor pada pemindaian otak: Segmentasi semantik (mengklasifikasikan setiap piksel untuk menentukan lokasi dan bentuk tumor) menggunakan CNN.
- Mengklasifikasikan artikel berita secara otomatis: Pemrosesan bahasa alami (NLP), khususnya klasifikasi teks menggunakan jaringan saraf rekurens (RNNs), CNNs, atau Transformers (lihat Bab 16).
- Menandai komentar ofensif secara otomatis di forum diskusi: Klasifikasi teks menggunakan alat NLP yang sama.
- Merangkum dokumen panjang secara otomatis: Ringkasan teks, cabang dari NLP yang menggunakan alat yang sama.
- Membuat chatbot atau asisten pribadi: Melibatkan komponen NLP yang beragam, termasuk pemahaman bahasa alami (NLU) dan sistem tanya-jawab.
- Memprediksi pendapatan perusahaan Anda tahun depan berdasarkan metrik kinerja: Tugas regresi menggunakan Regresi Linier atau Regresi Polinomial (Bab 4), mesin vektor dukungan regresi (SVMs; Bab 5), regresi Hutan Acak (Bab 7), atau jaringan saraf tiruan (Bab 10). Jika data melibatkan urutan, gunakan RNN, CNN, atau Transformers (Bab 15–16).
- Membuat aplikasi Anda merespons perintah suara: Pengenalan suara (pemrosesan sampel audio) menggunakan RNN, CNN, atau Transformers (Bab 15–16).
- Mendeteksi penipuan kartu kredit: Deteksi anomali (Bab 9).
- Mengelompokkan pelanggan berdasarkan pembelian mereka sehingga tim pemasaran Anda dapat merancang strategi yang ditargetkan: Pengelompokan (Bab 9).
- Menampilkan dataset kompleks dan berdimensi tinggi dalam diagram yang jelas dan informatif: Visualisasi data, sering melibatkan pengurangan dimensi (Bab 8).
- Merekomendasikan produk berdasarkan pembelian sebelumnya: Sistem rekomendasi yang memasukkan data ke dalam jaringan saraf tiruan (Bab 10).
- Membangun bot cerdas untuk permainan: Pembelajaran Penguatan (RL; Bab 18), di mana bot belajar untuk memaksimalkan imbalan (misalnya, AlphaGo mengalahkan juara dunia Go dengan menganalisis jutaan permainan dan bermain melawan dirinya sendiri).

Hal ini menunjukkan luasnya cakupan dan kompleksitas tugas-tugas ML.

## Jenis Sistem ML  
Sistem ML diklasifikasikan berdasarkan tiga kriteria (tidak eksklusif; mereka dapat digabungkan):
- Jumlah dan jenis pengawasan yang mereka terima selama pelatihan: Terawasi, tidak terawasi, semi-terawasi, Pembelajaran Penguatan.
- Apakah mereka dapat belajar secara bertahap secara real-time: Pembelajaran online versus pembelajaran batch.
- Apakah mereka bekerja dengan hanya membandingkan titik data baru dengan titik data yang sudah diketahui, atau dengan mendeteksi pola dalam data pelatihan dan membangun model prediktif: Pembelajaran berbasis instance versus pembelajaran berbasis model.

Misalnya, filter spam dapat menjadi sistem pembelajaran terawasi yang berbasis model dan belajar secara online.

### Pembelajaran Terawasi/Tidak Terawasi
- **Pembelajaran Terawasi**: Data pelatihan yang diberikan ke algoritma mencakup solusi yang diinginkan, yang disebut label (Gambar 1-5: Kumpulan data pelatihan berlabel untuk klasifikasi spam). Tugas umum:
<img width="978" height="420" alt="image" src="https://github.com/user-attachments/assets/2485d309-006c-49c4-a074-aa0d668d54be" />
  - Klasifikasi: Mengkategorikan contoh (misalnya, filter spam yang mengklasifikasikan email sebagai spam atau ham).
  - Regresi: Memprediksi nilai numerik (misalnya, memprediksi harga mobil berdasarkan prediktor seperti jarak tempuh, usia, merek; Gambar 1-6: Masalah regresi: memprediksi nilai berdasarkan fitur input). Catatan: Atribut adalah jenis data (misalnya, “jarak tempuh”), sementara fitur memiliki beberapa arti tetapi umumnya berarti atribut ditambah nilainya. Algoritma: k-Nearest Neighbors, Regresi Linier, Regresi Logistik (menghasilkan probabilitas kelas, misalnya 20% kemungkinan spam), Mesin Vektor Dukungan (SVM), Pohon Keputusan dan Hutan Acak, Jaringan Saraf Tiruan.
<img width="998" height="552" alt="image" src="https://github.com/user-attachments/assets/9eef420a-3666-43e2-b749-87f484eec813" />
 - Beberapa algoritma jaringan saraf (misalnya, autoencoders dan restricted Boltzmann machines) dapat bersifat tidak diawasi, sementara yang lain (misalnya, deep belief networks) bersifat semi-diawasi.

- **Unsupervised Learning**: Data pelatihan tidak dilabeli (Gambar 1-7: Pembelajaran tidak diawasi). Sistem belajar tanpa guru. Tugas utama (dibahas dalam Bab 8–9):
<img width="993" height="371" alt="image" src="https://github.com/user-attachments/assets/86ff9e2f-1be9-4d44-8e30-bded3acbe74b" />
  - Clustering: Mengelompokkan instance serupa secara otomatis (misalnya, mengelompokkan pengunjung blog berdasarkan demografi atau perilaku; Gambar 1-8: Clustering). Algoritma: K-Means, DBSCAN, Analisis Cluster Hierarkis (HCA).
<img width="981" height="362" alt="image" src="https://github.com/user-attachments/assets/a6c495e6-e6f4-4058-b538-847a4c440caa" />
  - Deteksi anomali dan deteksi keunikan: Mendeteksi insiden yang tidak biasa (misalnya, deteksi penipuan, cacat produksi; Gambar 1-10: Deteksi anomali). Algoritma: One-class SVM, Isolation Forest.
<img width="1008" height="417" alt="gambar" src="https://github.com/user-attachments/assets/6c9fb6e3-6f3c-4ab1-885d-5d8096113db2" /> 
  - Visualisasi dan pengurangan dimensi: Sederhanakan data tanpa kehilangan terlalu banyak informasi untuk visualisasi atau ekstraksi fitur (Gambar 1-9: Visualisasi dataset berdimensi tinggi menggunakan algoritma pengurangan dimensi; misalnya, t-SNE menyoroti cluster semantik seperti angka atau hewan). Algoritma: Principal Component Analysis (PCA), Kernel PCA, Locally Linear Embedding (LLE), t-Distributed Stochastic Neighbor Embedding (t-SNE). Ekstraksi fitur menggabungkan fitur yang berkorelasi (misalnya, jarak tempuh dan usia mobil menjadi “keausan”).
    <img width="984" height="668" alt="image" src="https://github.com/user-attachments/assets/9686c26e-7327-40fe-80a4-f125de0b6f08" />
  - Pembelajaran aturan asosiasi: Menemukan hubungan menarik antara atribut (misalnya, orang yang membeli saus barbekyu dan keripik kentang juga cenderung membeli steak). Algoritma: Apriori, Eclat.

- **Semisupervised Learning**: Menggunakan data pelatihan yang sebagian diberi label, biasanya terdiri dari banyak data yang tidak diberi label dan sedikit data yang diberi label (Gambar 1-11: Pembelajaran Semi-Supervised). Algoritma pertama-tama mengelompokkan instance yang serupa (tanpa pengawasan), lalu menggunakan label untuk menyebarkan ke dalam kelompok (misalnya, Google Photos mengenali orang yang sama dalam foto dengan hanya beberapa label). Contoh: Jaringan keyakinan dalam (DBN) berdasarkan mesin Boltzmann terbatas bertumpuk (RBM), dilatih tanpa pengawasan lalu disesuaikan dengan pengawasan.

<img width="979" height="477" alt="image" src="https://github.com/user-attachments/assets/6bd8cd47-7610-4ead-acce-53e4cd43856b" />


- **Pembelajaran Penguatan**: Sangat berbeda—sistem pembelajaran (agen) mengamati lingkungan, memilih dan melakukan tindakan, serta menerima hadiah atau hukuman (Gambar 1-12: Pembelajaran Penguatan). Ia belajar sendiri strategi terbaik (kebijakan) untuk mendapatkan hadiah maksimal seiring waktu (misalnya, robot belajar berjalan; AlphaGo belajar dengan menganalisis jutaan permainan dan bermain ribuan kali melawan dirinya sendiri; selama permainan melawan juara, proses belajar dimatikan).
<img width="986" height="511" alt="image" src="https://github.com/user-attachments/assets/5b6445a3-aa22-431f-bed2-3fa3449510ad" />

  <img width="998" height="692" alt="image" src="https://github.com/user-attachments/assets/d9e4a5ca-657d-40d3-bb56-67fa11c91c25" />

### Batch and Online Learning
- **Batch Learning (Offline Learning)**: Trained using all available data offline (takes time and resources). Then launched without further learning (applies what it learned). To handle new data or changes, train a new version from scratch on full dataset (including old and new data), then replace the old (can automate; Figure 1-3). Good if data doesn't change rapidly and resources allow.

- **Online Learning (Incremental Learning)**: Trained incrementally by feeding data instances sequentially, either individually or in mini-batches (Figure 1-13: Online learning). Fast and cheap, can learn on the fly from huge datasets or streaming data (e.g., predict stock prices). Out-of-core learning: Handles datasets too large for main memory by loading parts (Figure 1-14: Out-of-core learning). Learning rate hyperparameter: How fast to adapt (high: learns fast but forgets old data quickly; low: more inertia but less sensitive to noise/outliers or bad data). Challenge: If bad data is fed, performance declines—monitor and switch off learning or roll back if detected.

<img width="1018" height="511" alt="image" src="https://github.com/user-attachments/assets/a4d41b73-d3ca-48ab-aa76-bad734ec9def" />

<img width="980" height="519" alt="image" src="https://github.com/user-attachments/assets/282f9a6a-fe99-4968-8ab9-fd5aaa879412" />


### Instance-Based Versus Model-Based Learning
- **Instance-Based Learning**: The system learns the examples by heart, then generalizes to new cases using a similarity measure (e.g., flag an email as spam if very similar to known spam emails by word count; simplest form is k-Nearest Neighbors, classifying based on majority vote of most similar instances; Figure 1-15: Instance-based learning).

<img width="991" height="420" alt="image" src="https://github.com/user-attachments/assets/98b1da5e-2269-471d-9f57-c2d8f2b7b6bd" />

- **Model-Based Learning**: Builds a model of the examples, then uses that model to make predictions (Figure 1-16: Model-based learning). Example: Study if money makes people happy using life satisfaction and GDP per capita data—model life satisfaction as a linear function (Equation 1-1: life_satisfaction = θ₀ + θ₁ × GDP_per_capita). Training finds parameters θ₀ (bias) and θ₁ (weight) to fit the data best (minimizing cost function like mean squared error; Figures 1-17 to 1-19 show linear models fitting data). Utility function measures goodness, cost function badness.

<img width="1006" height="432" alt="image" src="https://github.com/user-attachments/assets/22ea5cdf-2573-4b76-b624-4a3576eb5d26" />

<img width="995" height="443" alt="image" src="https://github.com/user-attachments/assets/9e268587-fa31-4f88-b193-331e111dc4c9" />

<img width="991" height="433" alt="image" src="https://github.com/user-attachments/assets/0f037b69-8ff1-4e96-8cb1-e98d43c47f31" />

<img width="990" height="436" alt="image" src="https://github.com/user-attachments/assets/bbc7865f-25e6-4c9a-b83a-a9e9e5e04a4f" />


  To illustrate, the book provides a Python example using Scikit-Learn to train a linear regression model on life satisfaction vs. GDP data (Example 1-1). The code loads data, prepares it, visualizes a scatterplot, trains the model, and predicts for Cyprus:

  ```python:disable-run
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import LinearRegression

  # Download and prepare the data
  data_root = "https://github.com/ageron/data/raw/main/"
  lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
  X = lifesat[["GDP per capita (USD)"]].values
  y = lifesat[["Life satisfaction"]].values

  # Visualize the data
  lifesat.plot(kind='scatter', grid=True,
               x="GDP per capita (USD)", y="Life satisfaction")
  plt.axis([23_500, 62_500, 4, 9])
  plt.show()

  # Select a linear model
  model = LinearRegression()

  # Train the model
  model.fit(X, y)

  # Make a prediction for Cyprus
  X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
  print(model.predict(X_new))  # outputs [[6.30165767]]
  ```

  This code demonstrates loading data, visualization, model selection, training, and prediction. For comparison, a k-Nearest Neighbors regression (averaging neighbors' values) could be used instead.

## Typical Machine Learning Project Workflow
Study the data, explore and visualize it (e.g., scatterplot), prepare the data for ML algorithms, select and train models, fine-tune via error analysis and hyperparameter search, evaluate on test set, launch, monitor, and maintain. Chapter 2 covers an end-to-end project.

## Main Challenges of Machine Learning
Problems usually from bad data (quantity, quality) more than bad algorithms.

### Insufficient Quantity of Training Data

<img width="991" height="614" alt="image" src="https://github.com/user-attachments/assets/d6953c72-da4c-41a5-96c3-261aed72c2df" />

ML needs lots of data (thousands for simple problems, millions for complex like image or speech). 2001 paper "The Unreasonable Effectiveness of Data" notes algorithms perform similarly given enough data (Figure 1-20: Even mediocre algorithms can perform well with lots of data). For complex problems, data matters more than algorithms.

### Nonrepresentative Training Data
<img width="998" height="416" alt="image" src="https://github.com/user-attachments/assets/9af81e3b-863b-4d74-991f-ef35ad6addb4" />

To generalize well, training data must represent new cases. Sampling noise (chance in small samples) or sampling bias (flawed method, e.g., 1936 Literary Digest poll predicting Landon win due to telephone/wealthy bias; nonresponse bias). Example: Linear model on life satisfaction vs. GDP misses poor countries—adding them changes the model (Figure 1-21: Training data that is not representative).

### Poor-Quality Data
If full of errors, outliers, noise (poor measurements), hard to detect patterns. Spend time cleaning: Discard/fix outliers, decide on missing values (ignore attribute, ignore instances, fill with median/zero, or train one model with and one without the attribute).

### Irrelevant Features
Garbage in, garbage out. Feature engineering: Feature selection (select useful), feature extraction (combine to new, e.g., via dimensionality reduction), create new features by gathering new data.

### Overfitting the Training Data
<img width="1026" height="441" alt="image" src="https://github.com/user-attachments/assets/5b2d159d-494a-4871-9350-de1c6329fbf4" />

Model too complex, learns noise (poor generalization; Figure 1-22: Overfitting). Like high-degree polynomial fitting training data perfectly but useless for new. Solutions: Simplify (reduce parameters/features/degrees, regularization constraining complexity; Figure 1-23: Regularized models), gather more training data, reduce noise (fix errors, remove outliers).

<img width="1008" height="406" alt="image" src="https://github.com/user-attachments/assets/9df14404-6021-45d5-b430-9efede86ca3f" />


### Underfitting the Training Data
Opposite of overfitting—model too simple to learn structure. Solutions: Select powerful model (more parameters), feed better features, reduce constraints (e.g., reduce regularization hyperparameter).

## Testing and Validating
Estimate generalization error: Split data into training set (80% for small, less for big data) and test set (hold out). Train on training, evaluate on test (generalization error = error rate on new cases). If training error low but generalization high, overfitting.

### Hyperparameter Tuning and Model Selection
Holdout validation: Split training into smaller training set and validation set. Train multiple models with different hyperparameters on reduced training, select best on validation, train best model on full training, evaluate on test. If validation too small/imprecise, use cross-validation (train/evaluate multiple times on small validation sets, average; multiplies training time).

### Data Mismatch
E.g., scraped web images for mobile app—preprocess to match phone pics. Train on web data, but if poor performance on validation (phone pics), use train-dev set (hold out part of web training). If good on train-dev but poor on validation = data mismatch (preprocess more); if poor on train-dev = overfitting.

### No Free Lunch Theorem
No model better if nothing assumed about data (David Wolpert, 1996). In practice, make assumptions (e.g., linear for simple relations) and evaluate few reasonable models—no free lunch!

## Exercises
19 questions to test understanding (solutions in Appendix A):
1. How would you define Machine Learning?
2. Can you name four types of problems where it shines?
3. What is a labeled training set?
4. What are the two most common supervised tasks?
5. Can you name four common unsupervised tasks?
6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?
7. What type of algorithm would you use to segment your customers into multiple groups?
8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
9. What is an online learning system?
10. What is out-of-core learning?
11. What type of learning algorithm relies on a similarity measure to make predictions?
12. What is the difference between a model parameter and a learning algorithm’s hyperparameter?
13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?
14. Can you name four of the main challenges in Machine Learning?
15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?
16. What is a test set, and why would you want to use it?
17. What is the purpose of a validation set?
18. What is the train-dev set, when do you need it, and how do you use it?
19. What can go wrong if you tune hyperparameters using the test set?

This chapter sets the foundation; later chapters add hands-on code with Scikit-Learn, Keras, and TensorFlow.

```


