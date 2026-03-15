# HMM İle İzole Kelime Tanıma Sistemi 

Bu depo, YZM212 Makine Öğrenmesi dersi kapsamında HMM (Saklı Markov Modelleri) kullanılarak tasarlanan izole kelime tanıma sistemini içermektedir.

## Problem Tanımı
"EV" ve "OKUL" kelimelerini ses spektrumlarına (High/Low) göre sınıflandıracak bir modelin teorik (Viterbi) ve uygulamalı (Python/hmmlearn) olarak geliştirilmesidir.

## Veri
Modeller için fonemler "Gizli Durumlar" olarak ele alınmış, sesin frekans karakteristikleri ise "Gözlem Dizileri" (High, Low) olarak kullanılmıştır. 

## Yöntem
* **Teorik Kısım:** Gelen sesin en olası fonem dizisini adım adım hesaplamak için Viterbi Algoritması kullanılmıştır.
* **Pratik Kısım:** Python'da `hmmlearn` kütüphanesi kullanılarak her iki kelime için olasılık (Log-Likelihood) hesabı yapılmış ve karşılaştırılmıştır.

## Sonuçlar
`[High, Low]` gözlem dizisi teste sokulduğunda:
* Viterbi algoritması ile en olası fonem dizisi "e-v" olarak hesaplanmıştır.
* Python uygulamasında "EV" modelinin ürettiği puan, "OKUL" modelinden yüksek çıkmış ve kelime "EV" olarak sınıflandırılmıştır.

## Yorum ve Tartışma
* **Gürültü Etkisi:** Ses verisindeki gürültü, HMM modelindeki emisyon (yayılma) olasılıklarının dağılımını bozarak sistemin yanılma payını artırır.
* **Deep Learning Tercihi:** Binlerce kelimelik gerçek sistemlerde HMM'lerin durum geçişlerini modellemek ciddi hesaplama maliyeti yaratır. Derin öğrenme modelleri öznitelik çıkarımını otomatikleştirip çok daha büyük verilerle başa çıkabildiği için tercih edilmektedir.
