import numpy as np
from hmmlearn import hmm

# 1. 'EV' Modeli Tanımlaması
model_ev = hmm.CategoricalHMM(n_components=2)
model_ev.startprob_ = np.array([1.0, 0.0])
model_ev.transmat_ = np.array([[0.6, 0.4], [0.2, 0.8]])
model_ev.emissionprob_ = np.array([[0.7, 0.3], [0.1, 0.9]])

# 2. 'OKUL' Modeli Tanımlaması (Temsili 4 Durumlu Model)
model_okul = hmm.CategoricalHMM(n_components=4)
model_okul.startprob_ = np.array([1.0, 0.0, 0.0, 0.0])
model_okul.transmat_ = np.array([
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 1.0]
])
# Rastgele temsili emisyon (0:High, 1:Low)
model_okul.emissionprob_ = np.array([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])

# 3. Test Verisi [High, Low] -> [0, 1]
test_data = np.array([[0, 1]]).T

# 4. Log-Likelihood (Olasılık) Hesaplaması
score_ev = model_ev.score(test_data)
score_okul = model_okul.score(test_data)

print(f"EV Modeli Log-Likelihood: {score_ev:.4f}")
print(f"OKUL Modeli Log-Likelihood: {score_okul:.4f}")

if score_ev > score_okul:
    print("Siniflandirma Sonucu: EV")
else:
    print("Siniflandirma Sonucu: OKUL")
