# YZM212 — 4. Laboratuvar Ödevi: Uzak Bir Galaksinin Parlaklık Analizi

Bayesyen Çıkarım (MCMC) ile gürültülü astronomik gözlem verilerinden bir gök cisminin **gerçek parlaklığını (μ)** ve **ölçüm belirsizliğini (σ)** tahmin etme uygulaması.

## 1. Problem Tanımı

Uzak bir galaksiden gelen fotonları ölçen teleskobumuz atmosferik dalgalanmalar, ısıl gürültü ve toz etkileri yüzünden ham değil, **gürültülü** bir sinyal kaydeder. Görevimiz:

- Yalnız gözlem (`data`) verildiğinde, galaksinin **gerçek parlaklığı μ** hakkındaki inancı güncellemek,
- Aynı anda ölçüm sisteminin **gürültü seviyesi σ** hakkındaki inancı da çıkarmaktır.

Frekansçı bir nokta tahmin yerine, Bayesyen yaklaşım bize **olasılık dağılımı** (posterior) verir — belirsizliği açıkça raporlar.

Bayes Teoremi:

$$P(\theta\mid D) = \frac{P(D\mid\theta)\,P(\theta)}{P(D)}$$

- `θ = (μ, σ)` — tahmin edilecek parametreler
- `P(D|θ)` — **Likelihood:** verinin, parametreler doğruysa gözlemlenme olasılığı
- `P(θ)` — **Prior:** veriyi görmeden önceki inanç (fiziksel sınırlar vb.)
- `P(θ|D)` — **Posterior:** veriyi gördükten sonra güncellenmiş inanç

## 2. Veri

Sentetik gözlem verisi (doğa tarafından bilinen, biz tarafından bilinmeyen değerler):

| Değişken | Değer | Anlam |
|---|---|---|
| `true_mu` | **150.0** | Gerçek parlaklık (ground truth) |
| `true_sigma` | **10.0** | Gerçek ölçüm gürültüsü |
| `n_obs` | **50** | Gözlem sayısı |

```python
np.random.seed(42)
data = true_mu + true_sigma * np.random.randn(n_obs)
```

Bu örneklemin ampirik istatistikleri: `mean ≈ 147.75`, `std ≈ 9.24`. Örnekleme gürültüsü yüzünden ampirik ortalama 150.0'a eşit değildir — Bayesyen modelin bu sapmayı kapatıp kapatamayacağını sınayacağız.

## 3. Yöntem

`emcee` kütüphanesiyle **affine-invariant ensemble MCMC** örnekleyicisi kullanıldı.

- **Log-likelihood:** Normal dağılım — `−0.5·Σ[((d−μ)/σ)² + log(2πσ²)]`
- **Log-prior (geniş, informatif olmayan):** `0 < μ < 300`, `0 < σ < 50` içinde düz (uniform)
- **Örnekleyici:** 32 walker, 2000 adım, ilk 500 adım burn-in, thin=15

Üç senaryoyu karşılaştırıyoruz:

| Senaryo | Prior | n_obs | Amaç |
|---|---|---|---|
| **A** | geniş (0–300 / 0–50) | 50 | Ana simülasyon |
| **B** | **dar, yanlış:** μ ∈ [100, 110] | 50 | Prior etkisini göstermek |
| **C** | geniş | 5 | Veri miktarının belirsizliğe etkisi |

## 4. Sonuçlar

### 4.1 Parametre Karşılaştırma Tablosu (Senaryo A)

| Değişken | Gerçek Değer (Girdi) | Tahmin (Median) | Alt Sınır (%16) | Üst Sınır (%84) | Mutlak Hata |
|---|---:|---:|---:|---:|---:|
| **μ** (Parlaklık) | 150.00 | **147.65** | 146.36 | 149.01 | 2.35 |
| **σ** (Hata Payı) | 10.00 | **9.51** | 8.60 | 10.56 | 0.49 |

### 4.2 Prior Etkisi (Senaryo B)

| Değişken | Gerçek | Median | Alt (%16) | Üst (%84) | Mutlak Hata |
|---|---:|---:|---:|---:|---:|
| μ | 150.00 | **109.42** | 108.44 | 109.85 | **40.58** |
| σ | 10.00 | **40.04** | 36.44 | 44.22 | **30.04** |

Gerçek μ=150 prior aralığı [100,110]'un **dışında** olduğu için posterior bu aralığın tavanına sıkışır; model veriyi açıklamak için σ'yı şişirir — klasik *prior misspecification* örneği.

### 4.3 Veri Miktarı Etkisi (Senaryo C, n=5)

| Değişken | Gerçek | Median | Alt (%16) | Üst (%84) | Belirsizlik genişliği |
|---|---:|---:|---:|---:|---:|
| μ | 150.00 | 154.82 | 150.20 | 159.16 | ~±4.5 |
| σ | 10.00 | 9.21 | 6.22 | 15.12 | ~±4.5 |

Senaryo A'ya göre posterior genişlikleri yaklaşık **3 kat** artmıştır — `1/√n` kuralıyla uyumlu (`√(50/5) ≈ 3.16`).

## 5. Yorum ve Tartışma

### 5.1 Doğruluk (Accuracy)

Gürültü oranı ≈ %6.7 olmasına rağmen Senaryo A'da μ için mutlak hata ≈ 2.35 (%1.6), σ için ≈ 0.49 (%5). Tahmin **fiziksel olarak kabul edilebilir** seviyededir; hatanın büyük kısmı 50 kişilik örneklemin rastgele ortalama sapmasından gelir (ampirik ortalama 147.75). Posterior, ampirik ortalamanın etrafında konsantre olmuştur — yani model *veriye dürüsttür*, gerçeği sihirli bir şekilde "bilmez".

### 5.2 Hassasiyet (Precision) Karşılaştırması

μ posterior yarı-genişliği ≈ 1.3; σ'nınki ≈ 1.0. **Göreli** (fractional) hassasiyete bakılırsa μ %0.9, σ %10 hatalıdır. Ortalamanın **çok daha kesin** tahmin edilmesinin nedeni:

1. Ortalamanın standart hatası `σ/√n`, varyansınki `σ²·√(2/(n−1))` ile ölçeklenir.
2. Varyans/σ **ikinci moment** olduğundan aynı veri sayısıyla öğrenilmesi daha zordur.
3. n=50 "orta büyüklükte" bir örneklem: μ için yeterince dar posterior üretir ama σ için ek dalgalanma bırakır.

### 5.3 Korelasyon (Corner Plot Şekli)

Senaryo A'daki 2-B posterior yaklaşık **dairesel/dik eliptir**. Bu, Normal modelde μ ve σ'nın **bağımsız tahmin edildiğini** gösterir (sample mean ve sample variance, Normal dağılım altında istatistiksel olarak bağımsız yeterli istatistiklerdir). Bir parametrede hata yaptığımızda diğerini düzeltmek *gerekmiyor*.

### 5.4 Prior Etkisi

Senaryo B, Bayesyen çıkarımın en önemli uyarısını göstermektedir: **yanlış bir prior, sonsuz veriye rağmen sonucu bozar** (elbette burada kesin bir sınır kullanıldığı için olasılık tam olarak 0'dır — yumuşak bir prior olsa veri ağır basardı). Pratikte *informatif olmayan* (yeterince geniş) prior seçmek veya prior'un savunulabilir olduğuna emin olmak kritiktir.

### 5.5 Veri Miktarı

Senaryo C, Bayesyen çıkarımın en çekici özelliğini gösterir: *az veri ile bile* tahmin yapabiliriz, ancak **posterior bu azlığı geniş belirsizlikle raporlar**. Frekansçı yöntemler (örn. p-değerleri) böyle küçük örneklemlerde yanıltıcı güven iddiaları doğurabilirken, Bayesyen posterior dürüsttür.

## 6. Dosya Yapısı

```
4_odev_bayesian_parlaklik/
├── bayesian_parlaklik.ipynb       # Ana Jupyter defteri (çıktılarla birlikte)
├── run_simulation.py              # Aynı analizi script olarak çalıştırır
├── rapor.pdf                      # Grafikler + yorumlar (rapor)
├── README.md                      # Bu dosya
├── results.json                   # Sayısal sonuç özetleri
└── figures/
    ├── A_data_hist.png            # Sentetik gözlem verisi histogramı
    ├── A_trace.png                # MCMC zincirleri (yakınsama)
    ├── A_corner.png               # Senaryo A — Corner plot
    ├── B_corner_narrow_prior.png  # Senaryo B — dar prior
    ├── C_corner_small_n.png       # Senaryo C — n=5
    └── D_posterior_compare.png    # Üç senaryonun posteriorlarının karşılaştırması
```

## 7. Çalıştırma

```bash
pip install numpy matplotlib emcee corner
# Notebook olarak
jupyter notebook bayesian_parlaklik.ipynb
# veya script olarak
python run_simulation.py
```

## 8. Kaynaklar

- Ders materyali: 2025-2026 Bahar YZM212 — 4. Ödev tanımı
- Foreman-Mackey et al. 2013, *emcee: The MCMC Hammer*, PASP
- Foreman-Mackey, *corner.py*: https://corner.readthedocs.io
