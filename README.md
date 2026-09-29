# Practical Statistics for Data Scientists — Deep Learning Class

Reproduksi pembelajaran **Bab 1–4** dari Peter Bruce, Andrew Bruce, dan Peter Gedeck, *Practical Statistics for Data Scientists* (edisi 2, O’Reilly, 2020). Notebook menulis ulang contoh konsep dengan kode dan penjelasan orisinal; ini **bukan** salinan lengkap kode buku. Nama repo berakhiran `deep-learning` untuk mengikuti konteks mata kuliah, tetapi empat bab ini membahas fondasi statistik dan regresi, **bukan** semuanya model deep learning. Buku ini sendiri tidak membahas neural networks.

## Isi

| Bab | Notebook | Pokok materi dan hasil belajar |
|---|---|---|
| 1 — Exploratory Data Analysis | [Buka notebook](Chapter_01_Exploratory_Data_Analysis/notebook.ipynb) | Tipe data; mean, median, trimmed/weighted mean; SD, MAD, IQR, kuantil; histogram, boxplot, KDE; korelasi dan analisis dua variabel. Memahami pemilihan ringkasan sesuai distribusi dan unit analisis. |
| 2 — Data and Sampling Distributions | [Buka notebook](Chapter_02_Data_and_Sampling_Distributions/notebook.ipynb) | Sampling bias, distribusi sampling, CLT, standard error, bootstrap, confidence interval, QQ plot, dan keluarga distribusi. Membedakan variasi data individual dari ketidakpastian estimasi. |
| 3 — Statistical Experiments and Significance Testing | [Buka notebook](Chapter_03_Statistical_Experiments_and_Significance_Testing/notebook.ipynb) | A/B test, permutation test, nilai-p, Welch t-test, power, multiple testing, ANOVA, chi-square, dan simulasi multi-arm bandit. Menafsirkan uji dalam konteks desain eksperimen. |
| 4 — Regression and Prediction | [Buka notebook](Chapter_04_Regression_and_Prediction/notebook.ipynb) | Regresi sederhana/multiple, residual, train/test, cross-validation, interaksi dan dummy, ridge, polinomial, spline. Membedakan penjelasan koefisien dari kualitas prediksi data baru. |

Setiap notebook memuat ringkasan konsep, penurunan/rumus inti seperlunya, kode yang dapat dijalankan, interpretasi, dan keterbatasan. Output tersimpan setelah eksekusi. Penjelasan disusun dalam bahasa Indonesia untuk memudahkan pembelajaran.

## Menjalankan

Direkomendasikan Python 3.12. Di terminal dari root repo:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/execute_notebooks.py
```

Alternatif: buka notebook di VS Code/Jupyter atau Google Colab. Jika memakai Colab, unggah atau clone seluruh repo agar folder `data/` tersedia; kemudian jalankan sel berurutan. Simulasi memakai random seed tetap agar contoh bisa direproduksi. Hasil numerik dapat sedikit berbeda antarrilis paket/platform.

## Sumber dan batasan

- Acuan utama: Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical Statistics for Data Scientists*, 2nd ed. O’Reilly. Bab 1–4.
- Lima CSV kecil di `data/` diambil dari [repositori pendamping resmi](https://github.com/gedeck/practical-statistics-for-data-scientists/tree/master/data) milik penulis, bukan dibuat sendiri. Lisensi upstream GPL-3.0 dicantumkan dalam [LICENSE](LICENSE). Kode notebook di sini ditulis mandiri, dengan atribusi dataset dan buku.
- Bab 2 memakai populasi **sintetis** untuk simulasi; Bab 3 memakai reward **sintetis** untuk bandit; Bab 4 memakai dataset diabetes bawaan scikit-learn untuk contoh multivariabel. Ini adaptasi pembelajaran, bukan penggandaan seluruh dataset/contoh buku.
- Dataset contoh buku bersifat historis; jangan gunakan untuk klaim statistik terkini atau keputusan medis/produk nyata. Korelasi dan regresi pada data observasional tidak otomatis kausal.
- Cakupan sengaja dibatasi ke Bab 1–4 sesuai tenggat tugas, bukan seluruh buku.
