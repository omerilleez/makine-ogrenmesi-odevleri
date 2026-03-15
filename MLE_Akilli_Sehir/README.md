# MLE ile Akıllı Şehir Planlaması

## Problem Tanımı
Bir belediyenin ulaşım departmanı için, şehrin en yoğun ana caddesinden bir dakikada geçen araç sayısını gösteren veriler kullanılarak trafik yoğunluğu modeli çıkarılmak istenmektedir. Amacımız, verilerin Poisson Dağılımı'na uyduğu varsayımıyla, gelecekteki trafiği tahmin etmek için en uygun $\lambda$ parametresini Maximum Likelihood Estimation (MLE) yöntemiyle bulmaktır.

## Veri
1 dakikalık zaman dilimlerinde geçen araç sayıları (toplam 14 gözlem):
`[12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]`

## Yöntem
* **Analitik Yöntem:** Poisson dağılımı için Log-Likelihood fonksiyonu türetilmiş ve birinci türevi sıfıra eşitlenerek en iyi tahminin ( $\hat{\lambda}_{MLE}$ ) verilerin aritmetik ortalaması olduğu kanıtlanmıştır.
* **Sayısal Yöntem:** Python'da `scipy.optimize` kütüphanesi kullanılarak Negatif Log-Likelihood (NLL) fonksiyonu "Kayıp Fonksiyonu Minimizasyonu" mantığıyla minimize edilmiş ve sayısal $\lambda$ değeri bulunmuştur.

## Sonuçlar
* **Sayısal Tahmin (MLE lambda):** 12.1429
* **Analitik Tahmin (Ortalama):** 12.1429
Sayısal ve analitik tahminler birbiriyle tam uyuşmaktadır. Görselleştirme sonucunda elde edilen Poisson PMF grafiğinin, gerçek verilerin histogramıyla yüksek uyum sağladığı (iyi fit ettiği) gözlemlenmiştir.

## Yorum ve Tartışma (Outlier Analizi)
Veri setine hatalı olarak 200 gibi ekstrem bir aykırı değer (outlier) eklendiğinde, MLE formülü gereği (aritmetik ortalamaya bağlı olduğundan) $\lambda$ parametresi bu uç değerden doğrudan ve şiddetli biçimde etkilenir. Bu durum modelin gerçek yoğunluğu yansıtamamasına ve belediyenin gereksiz yol genişletme gibi hatalı planlamalar yapmasına neden olabilir. MLE yöntemi bu senaryoda aykırı değerlere karşı dayanıklı (robust) değildir.
