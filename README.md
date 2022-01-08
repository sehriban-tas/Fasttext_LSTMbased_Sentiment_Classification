# Fasttext_LSTMbased_Sentiment_Classification
## Metin Temsil Yöntemleri ve FastText Nedir?<br>
Kelime, cümlede kullanıma göre farklı anlamlar kazanabilir. Anlamsal bilginin çıkarılması metinlerin işlenmesinde önemlidir. Kelimelerin işlenebilir formattaki haline kelime temsili denir.Bu işlenebilir değerler sayısal değerlerdir. Verileri sayısal değerlere dönüştürmek için farklı yöntemler geliştirilmiştir. *FastText* de tahminleme yaklaşımları ile sözcükler arasındaki anlamsal ilişkileri çıkaran tahminleme tabanlı yöntemlerden birisidir. **FastText tek kelimeleri yapay sinir ağına girdi olarak vermek yerine kelimeleri birkaç harf bazlı “n-gram” halinde parçalar.**

## LSTM Nedir?<br>
vanishing gradient problemini çözmek için geliştirilmiştir .LSTM backprop’ta farklı zaman ve katmanlardan gelen hata değerini korumaya yarıyor. Daha sabit bir hata değeri sağlayarak recurrent ağların öğrenme adımlarının devam edebilmesini sağlamaktadır. Bunu sebep sonuç arasına yeni bir kanal açarak yapmaktadır. 
 


![lstm](https://user-images.githubusercontent.com/40441222/148655226-55e41608-6598-4850-8a7d-a9f12dd6348d.png)<br>
Standart RNN’ lerde, bu tekrarlanan modül, tek bir tanh katmanı gibi çok basit bir yapıya sahip olacaktır. LSTM birimi, uzun veya kısa zaman periyotlarını hatırlar. Bu kabiliyetin anahtarı, tekrarlanan bileşenlerinde hiçbir etkinleştirme işlevini kullanmamasıdır. Dolayısıyla, depolanan değer yinelemeli olarak değiştirilmez ve zaman içinde geri yayılımla eğitildiğinde eğim kaybolmaz

## Veri İşleme - Sınıflandırma <br>
TTC-3600 veri kümesi kullanılmıştır. Toplam 6 kategoride (ekonomi, kültür-sanat, sağlık, siyaset, spor, teknoloji) 600 doküman içermektedir.

![model](https://user-images.githubusercontent.com/40441222/148655400-378f5d34-4454-46cc-8140-3fae361f83c1.png)

Activasyon fonksiyonu olarak ‘softmax’ kullanılmış olup arkasından dense katmanına eklenmiştir. Dense, çoğu durumda çalışan standart bir katman türüdür. Yoğun bir katmanda, önceki katmandaki tüm düğümler mevcut katmandaki düğümlere bağlanır . EarlyStopping fonksiyonu ile model ezberlemeye başladığında eğitimin durdurulmasını sağlanır. 

![bb](https://user-images.githubusercontent.com/40441222/148655560-52686155-2e47-4b48-b4af-dc2d12aab286.jpg)

Model  %61  oranında başarı elde etmiştir. Modeli iyileştirmeye devam ederek sonuçları paylaşmaya çalışacağım.

![vv](https://user-images.githubusercontent.com/40441222/148655554-ad1e64e3-b07b-4de1-8203-5596dec91d13.png)
![t](https://user-images.githubusercontent.com/40441222/148655556-7b9c2ee7-ac68-4f48-8a70-c5b264be1112.png)
![tt](https://user-images.githubusercontent.com/40441222/148655557-369560b8-c1a6-45a7-89bf-05340575aa0a.png)
