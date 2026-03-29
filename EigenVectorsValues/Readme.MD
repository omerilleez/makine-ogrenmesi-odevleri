# Özdeğerler ve Özvektörler - Ödev Raporu

## Soru 1: Matris Manipülasyonu, Özdeğerler ve Özvektörlerin Makine Öğrenmesi ile İlişkisi

### Tanımlar

**Matris Manipülasyonu:** Matrisler üzerinde yapılan toplama, çarpma, transpoze alma, tersinir matris bulma gibi işlemlerin genel adıdır. Veriler genellikle matris formunda tutulduğu için makine öğrenmesinde sürekli matris işlemleri yapılır.

**Özdeğer (Eigenvalue):** Bir kare matris A için, `A * v = λ * v` denklemini sağlayan λ sayısına özdeğer denir. Yani matris bir vektöre uygulandığında, o vektörün sadece büyüklüğünü (ölçeğini) değiştirip yönünü değiştirmiyorsa, bu ölçek faktörü özdeğerdir.

**Özvektör (Eigenvector):** Yukarıdaki denklemdeki v vektörüne özvektör denir. Bu vektör, matris dönüşümü altında yönünü korur.

### Makine Öğrenmesi ile İlişki

Özdeğer ve özvektörler makine öğrenmesinde birçok yerde karşımıza çıkar:

1. **PCA (Temel Bileşen Analizi):** Boyut indirgeme için en çok kullanılan yöntemlerden biridir. Kovaryans matrisinin özdeğer ve özvektörleri hesaplanır. En büyük özdeğere sahip özvektörler, verideki en fazla varyansı açıklayan yönlerdir. Bu sayede yüksek boyutlu veri daha az boyuta indirgenebilir.

2. **Spektral Kümeleme:** Graf tabanlı kümeleme yöntemlerinde Laplacian matrisinin özvektörleri kullanılarak veriler kümelere ayrılır.

3. **Google PageRank:** Web sayfalarının önem sırasını belirlemek için geçiş matrisinin en büyük özdeğerine karşılık gelen özvektör kullanılır.

4. **Lineer Regresyon ve Regularizasyon:** Özellik matrisinin özdeğerleri, modelin kararlılığını ve çoklu doğrusallık (multicollinearity) problemini anlamak için kullanılır.

5. **SVD (Tekil Değer Ayrışımı):** Öneri sistemleri ve doğal dil işlemede kullanılan SVD, özdeğer ayrışımıyla yakından ilişkilidir.

6. **Markov Zincirleri:** Durağan durum dağılımını bulmak için geçiş matrisinin özvektörleri hesaplanır.

### Kaynaklar

- Brownlee, J. (2018). "Introduction to Matrices and Matrix Arithmetic for Machine Learning." Machine Learning Mastery. https://machinelearningmastery.com/introduction-to-matrices-and-matrix-arithmetic-for-machine-learning/
- Brownlee, J. (2019). "A Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning." Machine Learning Mastery. https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/
- Strang, G. (2016). "Introduction to Linear Algebra." 5th Edition, Wellesley-Cambridge Press.

---

## Soru 2: NumPy linalg.eig Fonksiyonunun İncelenmesi

### Fonksiyonun Kullanımı

`numpy.linalg.eig(a)` fonksiyonu kare bir matrisin özdeğerlerini ve özvektörlerini hesaplar.

```python
import numpy as np

A = np.array([[6, 1, -1],
              [0, 7, 0],
              [3, -1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)
```

**Parametreler:**
- `a`: Özdeğerleri hesaplanacak kare matris (n x n boyutunda)

**Dönüş Değerleri:**
- `eigenvalues`: Özdeğerleri içeren (n,) boyutlu dizi. Tekrar eden özdeğerler tekrar sayısı kadar yer alır. Sıralama garantisi yoktur.
- `eigenvectors`: Özvektörlerin sütun olarak dizildiği (n, n) boyutlu matris. `eigenvectors[:, i]` ifadesi `eigenvalues[i]`'ye karşılık gelen özvektördür.

### Arka Planda Nasıl Çalışır

NumPy'ın kaynak kodlarını incelediğimizde (`numpy/linalg/linalg.py`), `eig` fonksiyonunun şu adımları izlediğini görüyoruz:

1. **Girdi kontrolü:** Matrisin kare olup olmadığı kontrol edilir.
2. **LAPACK çağrısı:** Asıl hesaplama LAPACK (Linear Algebra PACKage) kütüphanesine devredilir. LAPACK, Fortran ile yazılmış yüksek performanslı bir lineer cebir kütüphanesidir.
   - Gerçel matrisler için `dgeev` rutini çağrılır
   - Kompleks matrisler için `zgeev` rutini çağrılır
3. **LAPACK algoritması:** İçeride QR algoritması kullanılır. Bu algoritma matrisi iteratif olarak üst üçgen (Schur) formuna dönüştürür ve özdeğerleri bu formdan çıkarır.
4. **Sonuç düzenleme:** LAPACK'ten dönen sonuçlar Python formatına dönüştürülür.

### Önemli Notlar

- Fonksiyon simetrik/Hermitian matrisler için optimize edilmemiştir. Simetrik matrisler için `numpy.linalg.eigh` kullanmak daha verimlidir.
- Sonuçlar her zaman karmaşık sayı olabilir (gerçel matrisler bile kompleks özdeğerlere sahip olabilir).
- Özvektörler normalize edilmiş halde döner (birim uzunlukta).

### Kaynak
- NumPy Dokümantasyon: https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
- NumPy Kaynak Kodları: https://github.com/numpy/numpy/tree/main/numpy/linalg

---

## Soru 3: Manuel ve NumPy Karşılaştırması

Bu soru için `EigenVectorsValues.ipynb` notebook dosyasında detaylı uygulama yapılmıştır.

### Kullanılan Yöntem

LucasBN'nin GitHub reposundaki yaklaşım temel alınmıştır. Manuel yöntemde şu adımlar izlenir:

1. **Karakteristik denklem oluşturma:** `A - λI` matrisi oluşturulur. Her eleman bir polinom haline gelir (örneğin köşegen elemanı 6 ise `6 - λ` yani `[6, -1]` polinomu).

2. **Determinant hesaplama:** `det(A - λI) = 0` denklemi, kofaktör açılımı ile polinom olarak hesaplanır. Bu polinoma karakteristik polinom denir.

3. **Kök bulma:** Karakteristik polinomun kökleri `np.roots()` ile bulunur. Bu kökler özdeğerlerdir.

### Sonuçlar

Test matrisi:
```
A = [[6, 1, -1],
     [0, 7, 0],
     [3, -1, 2]]
```

Her iki yöntem de aynı özdeğerleri hesaplamaktadır: **λ = 7, 5, 3**

### Farklar

| Özellik | Manuel Yöntem | NumPy eig |
|---------|--------------|-----------|
| Özdeğer hesaplama | Var | Var |
| Özvektör hesaplama | Yok | Var |
| Algoritma | Karakteristik polinom + kök bulma | QR Algoritması (LAPACK) |
| Performans | Büyük matrislerde yavaş | Optimize edilmiş, hızlı |
| Bağımlılık | np.roots kullanıyor | Tamamen LAPACK tabanlı |

Manuel yöntem eğitim amaçlı olarak mantığı anlamak için iyidir, ama gerçek projelerde NumPy'ın optimize edilmiş `eig` fonksiyonu tercih edilmelidir.

### Kaynak
- LucasBN. "Eigenvalues-and-Eigenvectors." GitHub. https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
