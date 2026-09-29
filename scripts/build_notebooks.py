"""Build original, educational notebooks for chapters 1–4.

The explanations and code here are independently written. The small CSV files
come from the book's companion repository; see README.md for attribution.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source)
        for kind, source in cells
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    path = ROOT / name / "notebook.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(path.relative_to(ROOT))


M = lambda text: ("md", text)
C = lambda text: ("code", text)


write("Chapter_01_Exploratory_Data_Analysis", [
    M("""# Bab 1 — Exploratory Data Analysis

**Rujukan:** Bruce, Bruce & Gedeck, *Practical Statistics for Data Scientists*, edisi 2, Bab 1. Notebook ini menulis ulang contoh inti dengan penjelasan dan kode orisinal, bukan salinan kode buku. Dataset `state.csv` berasal dari repositori pendamping resmi.

**Tujuan:** membedakan tipe data, menghitung ukuran pusat/sebaran, membaca distribusi dan hubungan antarvariabel, lalu menjelaskan mengapa ringkasan numerik saja tidak cukup. Jalankan dari root repositori atau melalui notebook ini (path dicari otomatis)."""),
    C("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
state = pd.read_csv(ROOT / 'data' / 'state.csv')
state.head()"""),
    M("""## 1. Struktur dan tipe data

Satu baris mewakili sebuah negara bagian AS. `Population` adalah hitungan, `Murder.Rate` variabel numerik kontinu, sedangkan `State` dan `Abbreviation` adalah label kategoris. Angka yang tampak seperti bilangan belum tentu layak dirata-ratakan: kode pos, misalnya, adalah identitas. Unit dan periode pengukuran harus diperiksa sebelum membandingkan kolom."""),
    C("""print(state.dtypes)
print('Ukuran data:', state.shape)
print('Nilai hilang per kolom:\\n', state.isna().sum())
assert state['State'].is_unique and len(state) == 50"""),
    M("""## 2. Ukuran lokasi

Rata-rata memakai semua nilai dan peka terhadap nilai ekstrem; median adalah titik tengah yang lebih tahan pencilan. Rata-rata terpangkas membuang sebagian observasi di kedua ekor sebelum menghitung rata-rata. Rata-rata tertimbang menghitung $\\bar{x}_w=\\sum_i w_i x_i/\\sum_i w_i$. Bobot penduduk di sini menjawab pertanyaan berbeda dari rata-rata negara bagian: gambaran untuk penduduk, bukan untuk negara bagian tipikal."""),
    C("""from scipy.stats import trim_mean

x = state['Population'].to_numpy()
location = pd.Series({
    'mean': x.mean(), 'median': np.median(x),
    'trimmed_10pct_each_tail': trim_mean(x, 0.10),
})
rate = state['Murder.Rate'].to_numpy()
weighted_rate = np.average(rate, weights=x)
print(location.round(1))
print('Rata-rata murder rate per state:', round(rate.mean(), 3))
print('Murder rate rata-rata berbobot populasi:', round(weighted_rate, 3))"""),
    M("""## 3. Ukuran variasi dan posisi

Varians sampel $s^2=\\sum(x_i-\\bar x)^2/(n-1)$ mengkuadratkan deviasi; simpangan baku kembali ke unit semula. MAD di bawah adalah *median absolute deviation* yang tidak diskalakan: median dari $|x_i-\\mathrm{median}(x)|$. IQR = kuartil 3 minus kuartil 1. MAD dan IQR lebih tahan terhadap pencilan. Pembagi $n-1$ relevan saat data dianggap sampel; 50 state di sini justru bisa dipandang populasi seluruh state AS pada dataset tersebut."""),
    C("""spread = pd.Series({
    'sample_std': np.std(x, ddof=1),
    'sample_variance': np.var(x, ddof=1),
    'unscaled_MAD': np.median(np.abs(x - np.median(x))),
    'IQR': np.quantile(x, .75) - np.quantile(x, .25),
    'range': x.max() - x.min(),
})
print(spread.round(1))
print('Kuantil populasi:\\n', state['Population'].quantile([0, .25, .5, .75, 1]))"""),
    M("""## 4. Distribusi

Histogram mengelompokkan observasi ke bin: bentuknya bergantung pada lebar bin. Boxplot menonjolkan median, IQR, dan kandidat pencilan (di luar 1,5×IQR), tetapi tidak menampilkan seluruh bentuk distribusi. Kurva KDE menghaluskan data dan hasilnya bergantung pada bandwidth. Pencilan bukan otomatis kesalahan; periksa konteks sebelum menghapusnya."""),
    C("""fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
state['Population'].hist(bins=10, ax=axes[0]); axes[0].set_title('Histogram populasi')
state.boxplot(column='Population', ax=axes[1]); axes[1].set_title('Boxplot populasi')
state['Murder.Rate'].plot.kde(ax=axes[2]); axes[2].set_title('KDE murder rate')
for ax in axes: ax.grid(alpha=.2)
plt.tight_layout(); plt.show()"""),
    M("""## 5. Dua variabel sekaligus

Korelasi Pearson mengukur asosiasi **linear**, dengan nilai dari -1 sampai 1. Nilainya tidak menyatakan sebab-akibat, dan dapat dipengaruhi pencilan. Scatterplot perlu dibaca bersama koefisiennya. Categorical summary di bawah adalah pengelompokan analitis buatan notebook, **bukan** kategori asli dalam dataset."""),
    C("""state['population_group'] = pd.qcut(state['Population'], q=4, labels=['Q1 kecil', 'Q2', 'Q3', 'Q4 besar'])
print('Korelasi Pearson:', round(state['Population'].corr(state['Murder.Rate']), 3))
print(state.groupby('population_group', observed=True)['Murder.Rate'].agg(['count', 'median', 'mean']).round(2))
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(state['Population'] / 1e6, state['Murder.Rate'], alpha=.7)
ax.set(xlabel='Populasi (juta)', ylabel='Murder rate', title='Hubungan dua variabel')
plt.tight_layout(); plt.show()"""),
    M("""### Ringkasan dan refleksi

- Pemilihan statistik harus mengikuti tipe data, unit analisis, dan pertanyaan riset.
- Mean/SD efisien untuk distribusi relatif simetris; median/IQR lebih tahan nilai ekstrem.
- Grafik adalah pemeriksaan terhadap ringkasan angka, bukan ornamen.
- Korelasi pada data observasional tidak membuktikan kausalitas. Data ini bersifat historis, sehingga tidak boleh ditafsirkan sebagai kondisi terkini."""),
])


write("Chapter_02_Data_and_Sampling_Distributions", [
    M("""# Bab 2 — Data and Sampling Distributions

**Rujukan:** buku utama edisi 2, Bab 2. Kode simulasi dan narasi ditulis mandiri. Tujuannya membedakan distribusi data dari distribusi statistik sampel, mengukur ketidakpastian dengan bootstrap, dan mengenali keluarga distribusi probabilitas. Simulasi sintetis diberi label jelas; ia bukan dataset penelitian nyata."""),
    C("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)
# Populasi sintetis miring ke kanan: 200.000 durasi kunjungan (detik)
population = rng.exponential(scale=60, size=200_000)
print('Rata-rata populasi simulasi:', round(population.mean(), 2))"""),
    M("""## 1. Sampel acak, bias seleksi, dan kesalahan sampling

Sampel acak memberi kesempatan terukur pada anggota populasi untuk terpilih. **Sampling error** adalah fluktuasi acak dari sampel ke sampel; **selection bias** muncul ketika mekanisme pemilihan sistematis mengubah representasi populasi. Memperbesar sampel tidak memperbaiki bias seleksi. Contoh `selected` sengaja memilih durasi lebih lama dan menunjukkan konsekuensinya, bukan meniru desain survei nyata."""),
    C("""random_sample = rng.choice(population, size=500, replace=False)
selected = rng.choice(population[population > 45], size=500, replace=False)
print(pd.Series({'population': population.mean(), 'random': random_sample.mean(),
                 'biased_selection': selected.mean()}).round(2))"""),
    M("""## 2. Distribusi sampling dan central limit theorem

Distribusi **data** memperlihatkan nilai individual. Distribusi **sampling** memperlihatkan satu statistik (misalnya mean) dari pengambilan sampel yang diulang. CLT menyatakan bahwa mean sampel mendekati distribusi normal dalam kondisi yang sesuai ketika $n$ membesar, meski data individual tidak normal. Standard error mean kira-kira $s/\\sqrt n$; ia mengukur ketidakpastian estimasi mean, bukan variasi antarindividu."""),
    C("""means_10 = np.array([rng.choice(population, 10).mean() for _ in range(1500)])
means_100 = np.array([rng.choice(population, 100).mean() for _ in range(1500)])
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, values, title in zip(axes, [population, means_10, means_100],
                              ['Data individual', 'Mean sampel n=10', 'Mean sampel n=100']):
    ax.hist(values, bins=35, density=True); ax.set_title(title); ax.set_xlim(0, 300)
plt.tight_layout(); plt.show()
print('SD mean n=10:', round(means_10.std(ddof=1), 2), 'n=100:', round(means_100.std(ddof=1), 2))"""),
    M("""## 3. Bootstrap dan interval kepercayaan

Bootstrap mengambil sampel **dengan pengembalian dari sampel yang tersedia**, lalu menghitung ulang statistik. Persentil 2,5 dan 97,5 dari statistik bootstrap membentuk interval persentil 95%. Interpretasi frequentist: prosedur interval akan mencakup parameter sejati kira-kira 95% kali dalam pengulangan eksperimen yang memenuhi asumsi. Bukan berarti ada peluang 95% parameter tetap itu berada dalam interval yang sudah dihitung. Bootstrap tidak menghilangkan bias pada sampel awal."""),
    C("""sample = rng.choice(population, size=200, replace=False)
boot_means = np.array([rng.choice(sample, len(sample), replace=True).mean() for _ in range(4000)])
ci = np.quantile(boot_means, [.025, .975])
print('Mean sampel:', round(sample.mean(), 2))
print('Bootstrap SE:', round(boot_means.std(ddof=1), 2))
print('95% percentile CI:', ci.round(2))
assert ci[0] < sample.mean() < ci[1]"""),
    M("""## 4. Distribusi teoritis dan QQ plot

Normal memiliki bentuk simetris; distribusi $t$ berekor lebih tebal (khususnya df kecil), relevan untuk inferensi mean dengan varians tak diketahui. QQ plot membandingkan kuantil data dengan normal teoretis. Titik yang membelok dari garis, terutama di ekor, menunjukkan bahwa normalitas mungkin tidak cocok. Uji/plot tidak menggantikan pengetahuan mekanisme data."""),
    C("""fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
z = np.linspace(-4, 4, 300)
axes[0].plot(z, stats.norm.pdf(z), label='Normal')
axes[0].plot(z, stats.t.pdf(z, df=3), label='t, df=3')
axes[0].legend(); axes[0].set_title('Kerapatan probabilitas')
stats.probplot(sample, dist='norm', plot=axes[1]); axes[1].set_title('QQ plot sampel eksponensial')
plt.tight_layout(); plt.show()"""),
    M("""## 5. Binomial, chi-square, F, Poisson, eksponensial, Weibull

Binomial menghitung keberhasilan dalam $n$ percobaan Bernoulli independen dengan peluang tetap $p$. Poisson memodelkan hitungan per interval dengan laju tetap; eksponensial memodelkan waktu tunggu antarperistiwa Poisson. Weibull memperluas model waktu tunggu dengan hazard yang bisa berubah. Chi-square dan F banyak muncul sebagai distribusi statistik uji; pilihan model selalu membutuhkan pemeriksaan asumsi."""),
    C("""summary = pd.DataFrame({
    'distribution': ['Binomial n=20,p=.3', 'Poisson mu=3', 'Exponential scale=2', 'Weibull shape=1.5'],
    'illustrative_mean': [rng.binomial(20, .3, 10000).mean(), rng.poisson(3, 10000).mean(),
                          rng.exponential(2, 10000).mean(),
                          (2 * rng.weibull(1.5, 10000)).mean()],
})
print(summary.round(3).to_string(index=False))
print('P(X >= 8), X~Binomial(20,.3):', round(stats.binom.sf(7, 20, .3), 4))
print('95% quantile chi-square(df=4):', round(stats.chi2.ppf(.95, 4), 3))
print('95% quantile F(df1=3, df2=20):', round(stats.f.ppf(.95, 3, 20), 3))"""),
    M("""### Ringkasan dan refleksi

- Representativitas ditentukan oleh desain sampling, bukan ukuran sampel saja.
- Standard error mengecil saat ukuran sampel bertambah, tetapi hanya menangani ketidakpastian sampling.
- Bootstrap berguna untuk statistik yang distribusi sampling-nya rumit; tetap harus menghormati struktur data (misalnya data berkelompok/time series).
- Distribusi probabilitas adalah model bersyarat pada asumsi, bukan label otomatis untuk semua data."""),
])


write("Chapter_03_Statistical_Experiments_and_Significance_Testing", [
    M("""# Bab 3 — Statistical Experiments and Significance Testing

**Rujukan:** buku utama edisi 2, Bab 3. Dataset `web_page_data.csv`, `four_sessions.csv`, dan `click_rates.csv` dari repositori pendamping resmi. Kode dan uraian merupakan implementasi mandiri. Nilai-p menjawab pertanyaan tentang data *jika hipotesis nol benar*, bukan probabilitas hipotesis nol benar."""),
    C("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
rng = np.random.default_rng(42)
web = pd.read_csv(ROOT / 'data' / 'web_page_data.csv')
web.groupby('Page')['Time'].agg(['count', 'mean', 'median']).round(3)"""),
    M("""## 1. A/B test, hipotesis dan randomisasi

Dalam A/B test, unit eksperimen dialokasikan acak ke perlakuan A/B. Hipotesis nol $H_0$ dapat berupa tidak ada perbedaan mean; hipotesis alternatif menyatakan ada perbedaan. Randomisasi membantu memisahkan efek perlakuan dari karakteristik unit, asalkan eksperimentasi dilakukan dengan baik. Dataset kecil ini dipakai untuk mendemonstrasikan perhitungan, bukan untuk membuat keputusan produk."""),
    C("""a = web.loc[web.Page == 'Page A', 'Time'].to_numpy()
b = web.loc[web.Page == 'Page B', 'Time'].to_numpy()
observed = b.mean() - a.mean()
print('Mean B - A:', round(observed, 4), 'detik')
assert len(a) > 0 and len(b) > 0"""),
    M("""## 2. Permutation test dan nilai-p

Di bawah $H_0$ bahwa label A/B dapat dipertukarkan, acak ulang label sambil mempertahankan ukuran grup. Statistik uji adalah selisih mean. Nilai-p dua sisi Monte Carlo dihitung sebagai proporsi permutasi dengan $|T^*|\\ge|T_{obs}|$, memakai koreksi +1 agar tidak tepat nol. Kesimpulan bergantung pada desain dan asumsi *exchangeability*. Efek dan interval kepercayaan tetap penting meskipun nilai-p kecil."""),
    C("""combined = np.concatenate([a, b])
permuted = np.empty(5000)
for i in range(len(permuted)):
    shuffle = rng.permutation(combined)
    permuted[i] = shuffle[len(a):].mean() - shuffle[:len(a)].mean()
p_perm = (1 + np.count_nonzero(np.abs(permuted) >= abs(observed))) / (len(permuted) + 1)
print('Permutation p-value (two-sided):', round(p_perm, 4))
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(permuted, bins=35); ax.axvline(observed, color='red', label='observed')
ax.axvline(-observed, color='red', linestyle='--'); ax.legend(); plt.show()"""),
    M("""## 3. t-test, Type I/II error, dan power

Welch t-test membandingkan mean dua grup tanpa mengasumsikan varians sama. Type I error: menolak $H_0$ yang benar; Type II: gagal menolak $H_0$ yang salah. Power = peluang mendeteksi efek tertentu dengan desain dan aturan uji yang ditentukan. Perkiraan $n$ di bawah adalah pendekatan normal dua-sampel dengan ukuran grup sama, bukan perhitungan desain lengkap."""),
    C("""t_result = stats.ttest_ind(a, b, equal_var=False)
print('Welch t statistic:', round(t_result.statistic, 3), 'p:', round(t_result.pvalue, 4))
alpha, target_power, effect, sigma = .05, .80, .25, 1.0
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(target_power)
n_per_group = np.ceil(2 * (z_alpha + z_beta)**2 * sigma**2 / effect**2)
print('Approximate n/group for standardized effect 0.25:', int(n_per_group))"""),
    M("""## 4. Multipel pengujian

Jika banyak hipotesis diuji pada ambang 0,05, peluang setidaknya satu false positive meningkat. Bonferroni mengontrol family-wise error rate dengan ambang $\\alpha/m$. Benjamini–Hochberg (BH) menargetkan false discovery rate dalam kondisi yang sesuai. Nilai-p contoh di bawah hanya ilustrasi mekanik; tidak berasal dari lima eksperimen nyata."""),
    C("""p_values = np.array([.002, .018, .04, .11, .72])
m = len(p_values)
order = np.argsort(p_values)
ranked = p_values[order]
bonferroni = p_values <= .05 / m
bh_cutoff = np.arange(1, m+1) * .05 / m
valid = np.flatnonzero(ranked <= bh_cutoff)
bh = np.zeros(m, dtype=bool)
if len(valid): bh[order[:valid[-1]+1]] = True
print(pd.DataFrame({'p': p_values, 'Bonferroni': bonferroni, 'BH': bh}))"""),
    M("""## 5. ANOVA dan chi-square

ANOVA satu arah menguji apakah semua mean kelompok sama, dengan rasio variasi antar- dan dalam-kelompok. Uji F memerlukan perhatian pada independensi, bentuk residual, dan homogenitas varians. Untuk data hitungan kategoris, chi-square menguji independensi pada tabel kontingensi; frekuensi harapan kecil dapat membuat pendekatan asimtotik buruk, sehingga prosedur exact/alternatif dibutuhkan. Hasil signifikan tidak menunjukkan pasangan mana yang berbeda tanpa analisis lanjutan."""),
    C("""sessions = pd.read_csv(ROOT / 'data' / 'four_sessions.csv')
groups = [group['Time'].to_numpy() for _, group in sessions.groupby('Page')]
anova = stats.f_oneway(*groups)
print('ANOVA F:', round(anova.statistic, 3), 'p:', round(anova.pvalue, 4))
clicks = pd.read_csv(ROOT / 'data' / 'click_rates.csv')
table = clicks.pivot(index='Headline', columns='Click', values='Rate')
chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
print(table)
print('Chi-square:', round(chi2, 3), 'p:', round(p_chi2, 4), 'min expected:', round(expected.min(), 2))"""),
    M("""## 6. Multi-arm bandit (simulasi tambahan)

Bandit mengalokasikan lebih banyak traffic ke opsi yang tampak baik sambil terus mengeksplorasi. Kebijakan epsilon-greedy memilih arm acak dengan peluang $\\epsilon$, selebihnya arm dengan rata-rata reward tertinggi saat ini. Ini berbeda dari A/B test tetap: alokasi adaptif mengubah analisis inferensinya. Angka reward di bawah **sintetis**, bukan data buku."""),
    C("""true_ctr = np.array([.04, .06, .05])
count = np.zeros(3, dtype=int); reward = np.zeros(3, dtype=int)
for _ in range(3000):
    arm = rng.integers(3) if rng.random() < .1 or not np.all(count) else np.argmax(reward / count)
    count[arm] += 1
    reward[arm] += rng.random() < true_ctr[arm]
print(pd.DataFrame({'true_CTR': true_ctr, 'visits': count, 'observed_CTR': reward/count}))"""),
    M("""### Ringkasan dan refleksi

- Rancang randomisasi, ukuran sampel, outcome utama, dan aturan berhenti sebelum melihat hasil.
- Nilai-p bukan ukuran efek, probabilitas $H_0$, maupun jaminan replikasi.
- Banyak uji membutuhkan kontrol kesalahan; keputusan sebaiknya memakai ukuran efek dan ketidakpastian.
- Bandit mengoptimalkan alokasi reward, sedangkan eksperimen klasik terutama mengestimasi efek; keduanya punya tujuan berbeda."""),
])


write("Chapter_04_Regression_and_Prediction", [
    M("""# Bab 4 — Regression and Prediction

**Rujukan:** buku utama edisi 2, Bab 4. `LungDisease.csv` dari repositori pendamping dipakai untuk regresi sederhana. Untuk memperlihatkan banyak prediktor, interaksi dan spline tanpa dataset eksternal besar, bagian lanjutan memakai dataset diabetes yang dibundel scikit-learn. Kode dan uraian ditulis mandiri. Tidak ada neural network di bab ini."""),
    C("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
lung = pd.read_csv(ROOT / 'data' / 'LungDisease.csv')
lung.describe().round(2)"""),
    M("""## 1. Regresi linear sederhana

Model $y_i=\\beta_0+\\beta_1x_i+\\epsilon_i$ menghubungkan paparan (`Exposure`) dengan fungsi paru (`PEFR`). Ordinary least squares memilih koefisien yang meminimalkan jumlah kuadrat residual. Slope menyatakan perubahan prediksi outcome untuk kenaikan satu unit prediktor, **bukan** efek kausal tanpa desain dan asumsi tambahan. Hubungan bisa dipengaruhi confounder yang tidak dicatat."""),
    C("""simple = LinearRegression().fit(lung[['Exposure']], lung['PEFR'])
print('Intercept:', round(simple.intercept_, 2), 'slope:', round(simple.coef_[0], 3))
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(lung['Exposure'], lung['PEFR'], alpha=.6)
xx = np.linspace(lung.Exposure.min(), lung.Exposure.max(), 100)
ax.plot(xx, simple.predict(pd.DataFrame({'Exposure': xx})), color='red')
ax.set(xlabel='Exposure', ylabel='PEFR', title='Regresi sederhana: prediksi, bukan kausalitas')
plt.tight_layout(); plt.show()"""),
    M("""## 2. Residual dan diagnosis awal

Residual $e_i=y_i-\\hat y_i$. Pola melengkung mengisyaratkan hubungan tidak linear; bentuk kipas dapat menunjukkan heteroskedastisitas; titik jauh bisa berpengaruh besar. Plot residual bukan uji final, tetapi pemeriksaan awal yang wajib. Ekstrapolasi di luar rentang paparan yang diamati memiliki ketidakpastian lebih tinggi."""),
    C("""fitted = simple.predict(lung[['Exposure']])
residual = lung['PEFR'].to_numpy() - fitted
print('RMSE in-sample:', round(root_mean_squared_error(lung['PEFR'], fitted), 2))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.scatter(fitted, residual, alpha=.6); ax.axhline(0, color='red')
ax.set(xlabel='Prediksi PEFR', ylabel='Residual', title='Residual vs fitted')
plt.tight_layout(); plt.show()"""),
    M("""## 3. Regresi multiple dan validasi di data tak terlihat

Beberapa prediktor memungkinkan estimasi asosiasi *kondisional*: koefisien satu fitur dibaca dengan fitur lain tetap. Multikolinearitas dapat membuat koefisien tidak stabil. Train/test split memisahkan evaluasi dari fitting; cross-validation mengulang pembagian agar estimasi kinerja lebih stabil. Model, transformasi, dan pilihan fitur seharusnya dipilih hanya dari training/CV; test set disimpan sebagai pemeriksaan akhir. Metrik MAE dan RMSE berada dalam unit outcome; RMSE memberi penalti lebih besar pada error besar."""),
    C("""diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
ols = LinearRegression().fit(X_train, y_train)
pred = ols.predict(X_test)
print(pd.Series({'test_MAE': mean_absolute_error(y_test, pred),
                 'test_RMSE': root_mean_squared_error(y_test, pred),
                 'test_R2': r2_score(y_test, pred)}).round(3))
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(LinearRegression(), X_train, y_train, cv=cv,
                        scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'})
print('CV MAE:', round(-scores['test_mae'].mean(), 2), 'CV R2:', round(scores['test_r2'].mean(), 3))"""),
    M("""## 4. Variabel kategori dan interaksi

Kategori harus diubah menjadi indikator (*dummy*) dengan satu level referensi untuk menghindari ketergantungan linear sempurna bersama intercept. Interaksi memungkinkan slope satu prediktor berubah menurut prediktor lain. Di sini kelompok BMI dibuat **khusus demonstrasi** dari median training; ambang yang dipelajari pada seluruh data sebelum split akan menyebabkan leakage. Contoh ini bukan model medis yang siap pakai."""),
    C("""threshold = X_train['bmi'].median()
def design(frame):
    out = frame[['age', 'bmi', 'bp']].copy()
    out['high_bmi'] = (frame['bmi'] > threshold).astype(int)
    out['age_x_high_bmi'] = out['age'] * out['high_bmi']
    return out

interaction = LinearRegression().fit(design(X_train), y_train)
print('Interaction model test RMSE:', round(root_mean_squared_error(y_test, interaction.predict(design(X_test))), 2))
print(pd.Series(interaction.coef_, index=design(X_train).columns).round(2))"""),
    M("""## 5. Regularisasi, model selection, dan bias–variance

Menambah prediktor selalu bisa menurunkan error training OLS, tetapi tidak menjamin prediksi baru lebih baik. Ridge menambah penalti $\\lambda\\sum_j\\beta_j^2$ sehingga koefisien menyusut; standardisasi dimasukkan **di dalam pipeline** agar tiap fold CV mempelajari parameter scaler dari training fold saja. Grid kecil ini dipilih dengan CV training; metrik test dihitung sekali setelah pemilihan."""),
    C("""candidates = [0.1, 1.0, 10.0, 100.0]
rows = []
for alpha in candidates:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    cv_result = cross_validate(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
    rows.append({'alpha': alpha, 'CV_RMSE': -cv_result['test_score'].mean()})
selection = pd.DataFrame(rows).sort_values('CV_RMSE')
best_alpha = float(selection.iloc[0]['alpha'])
best = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(X_train, y_train)
print(selection.round(2).to_string(index=False))
print('Selected Ridge test RMSE:', round(root_mean_squared_error(y_test, best.predict(X_test)), 2))"""),
    M("""## 6. Kurva non-linear: polinomial dan spline

Regresi masih bisa linear dalam **koefisien** walaupun fitur adalah fungsi non-linear dari $x$. Polinomial derajat tinggi mudah berosilasi di batas. Spline membagi rentang $x$ menjadi beberapa bagian polinomial halus. Keduanya perlu memilih kompleksitas lewat validasi; grafik berikut hanya eksplorasi *in-sample* pada hubungan BMI–outcome, bukan bukti bahwa model terbaik untuk prediksi."""),
    C("""bmi = X_train[['bmi']]
curves = {
    'linear': LinearRegression(),
    'quadratic': make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()),
    'spline': make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression()),
}
grid = pd.DataFrame({'bmi': np.linspace(bmi.bmi.min(), bmi.bmi.max(), 200)})
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(bmi.bmi, y_train, s=12, alpha=.35, label='training data')
for label, model in curves.items():
    model.fit(bmi, y_train)
    ax.plot(grid.bmi, model.predict(grid), label=label)
ax.set(xlabel='BMI (scaled)', ylabel='Disease progression', title='Perbandingan bentuk hubungan')
ax.legend(); plt.tight_layout(); plt.show()"""),
    M("""### Ringkasan dan refleksi

- Bedakan tujuan **prediksi** (kinerja pada data baru) dari **inferensi** (interpretasi koefisien/asumsi).
- Periksa residual, titik berpengaruh, non-linearitas, heteroskedastisitas, dan kemungkinan confounding.
- CV untuk memilih model; test set hanya untuk evaluasi final, dengan semua transformasi dilatih di training fold.
- Bab ini memakai model statistik klasik, bukan deep learning. Penamaan repo mengikuti konteks kelas."""),
])
