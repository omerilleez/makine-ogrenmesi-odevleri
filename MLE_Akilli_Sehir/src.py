# Bağımlılıklar
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ==========================================
# Bölüm 2: Python ile Sayısal (Numerical) MLE
# ==========================================

# Gözlemlenen Trafik Verisi (1 dakikada geçen araç sayısı)
traffic_data = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])

def negative_log_likelihood(lam, data):
    """
    Poisson daglimi için Negatif Log-Likelihood hesaplar.
    İpucu: log(k!) terimi optimizasyon sirasinda sabit oldugu icin ihmal edilebilir.
    """
    n = len(data)
    # Log-likelihood formülünün negatif hali (NLL)
    # Sabit olan log(k!) kısmını dahil etmiyoruz.
    nll = - (-n * lam + np.log(lam) * np.sum(data))
    return nll

# Başlangıç tahmini
initial_guess = np.array([1.0])

# Optimizasyon: NLL'yi minimize etmek, Likelihood'u maximize etmektir.
# bounds parametresi lambda'nın 0'dan büyük olmasını sağlar.
result = opt.minimize(negative_log_likelihood, initial_guess, args=(traffic_data,), bounds=[(0.001, None)])

mle_lambda = result.x[0]
analytic_lambda = np.mean(traffic_data)

print(f"Sayisal Tahmin (MLE lambda): {mle_lambda:.4f}")
print(f"Analitik Tahmin (Ortalama): {analytic_lambda:.4f}")

# ==========================================
# Bölüm 3: Model Karşılaştırma ve Görselleştirme
# ==========================================

# Olası k değerleri (x ekseni için)
k_values = np.arange(0, np.max(traffic_data) + 5)
# Bulunan lambda değeri ile Poisson PMF hesaplaması
poisson_pmf = poisson.pmf(k_values, mle_lambda)

# Görselleştirme
plt.figure(figsize=(10, 6))

# Gerçek verinin histogramı
plt.hist(traffic_data, bins=np.arange(min(traffic_data)-0.5, max(traffic_data)+1.5, 1), 
         density=True, alpha=0.6, color='blue', edgecolor='black', label='Gerçek Veri Histogrami')

# MLE ile bulunan Poisson Dağılımı
plt.plot(k_values, poisson_pmf, 'ro-', ms=8, label=f'Poisson PMF (λ={mle_lambda:.2f})')
plt.vlines(k_values, 0, poisson_pmf, colors='r', lw=2, alpha=0.5)

# Grafik etiketleri ve başlık
plt.xlabel('Arac Sayisi (1 dakikada)')
plt.ylabel('Olasilik / Frekans')
plt.title('Sehir Trafigi: Gercek Veri ve Poisson Modeli Karsilastirmasi')
plt.legend()
plt.grid(axis='y', alpha=0.75)

# Grafiği göster
plt.savefig('grafik.png', bbox_inches='tight')
plt.show()