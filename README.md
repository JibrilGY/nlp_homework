# NLP Homework: English Language Models

PG-19 verisinin bir alt kümesi üzerinde bigram, trigram ve kelime düzeyinde LSTM dil modelleri. Streamlit uygulaması başlangıç metninden devam üretir ve kayıtlı test sonuçlarını gösterir.

## Gereksinimler

- Test edilen Python sürümü: **3.12.3**.
- PyTorch **2.15.0**, Streamlit **1.66.1**, ipykernel **7.4.0**.
- Eğitim ortamı: NVIDIA GeForce RTX 3060 Laptop GPU; PyTorch CUDA 13.2 kurulumu.
- Uygulama ve değerlendirme, GPU bulunmadığında CPU kullanır. Eğitim notebook'undaki `device = torch.device("cuda")` satırı CPU eğitimi için `torch.device("cpu")` olarak değiştirilmelidir.

## Kurulum ve uygulamayı çalıştırma

Komutları projenin kök klasöründe çalıştırın:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

CPU kurulumu için önce CPU paketini yükleyip ardından proje bağımlılıklarını kurabilirsiniz:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU kurulumu için sistemin NVIDIA sürücüsüne uygun paket seçilmelidir: [PyTorch kurulum rehberi](https://pytorch.org/get-started/locally/).

Uygulama `artifacts/bigram.pkl`, `artifacts/trigram.pkl` ve `artifacts/lstm_best.pt` dosyalarını açılışta bir kez yükler. Arayüzden model, başlangıç metni, üretim uzunluğu, top-k ve seed seçilir. Noktalama işaretleri de token sayısına dahildir. LSTM üretiminde temperature **1.0** olarak sabittir.

## Veri kaynağı ve hazırlama

- [Kaggle: PG-19 Language Modeling Benchmark](https://www.kaggle.com/datasets/tunguz/the-pg19-language-modeling-benchmark-dataset)
- [Orijinal PG-19 deposu](https://github.com/google-deepmind/pg19)
- [İlgili çalışma: Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)

Orijinal train/validation/test ayrımı korunmuştur. Her bölümde kitap dosyaları önce sıralanır, ardından `SEED=42` ile karıştırılır. Token sınırına ulaşılınca seçim durur; son kitaptan yalnızca gerekli başlangıç bölümü alınabilir.

| Bölüm | Kitap/parça sayısı | Token sınırı |
|---|---:|---:|
| Train | 20 | 1.000.000 |
| Validation | 1 | 100.000 |
| Test | 3 | 100.000 |

Çizgili tablo ve dekoratif ayırıcı satırları temizlenir. Kıvrık apostroflar normalize edilir, harfler küçültülür ve düzenli ifadeyle kelime/noktalama tokenizasyonu uygulanır. Stopword silme veya stemming uygulanmaz. Sözlük yalnızca eğitim verisinden oluşturulur: minimum sıklık **2**, özel token'lar dahil maksimum boyut **20.000**. Sözlük dışı token'lar `<unk>` ile temsil edilir.

Seçilen kitap kimlikleri, kullanılan token sayıları ve işlenmiş dosyaların SHA-256 değerleri `results/data_selection.json` içinde kayıtlıdır. Validation tek kitaptan, test üç kitaptan oluşur. Hazırlanan alt küme ve tokenizasyon nedeniyle sonuçlar tam PG-19 benchmark sonuçlarıyla doğrudan karşılaştırılamaz.

## Ham veri yolu

`config.py`, `PG19_DATA_DIR` ortam değişkenini kullanır. Değişken tanımlı değilse önce `data/pg19/`, bu klasör yoksa kullanıcının `Desktop/dataset/` klasörü seçilir.

```bash
export PG19_DATA_DIR="/verinin/bulundugu/pg19"
```

Hem `train/*.txt` hem de Kaggle arşivindeki `train/train/*.txt` yapısı desteklenir. Aynı kural `validation` ve `test` için de geçerlidir. Ortam değişkenini notebook kernel'ini başlatmadan önce tanımlayın; PyCharm kullanıyorsanız kernel/çalıştırma ortamına ekleyin.

## Notebook sırası

Notebook'ları proje kökünden, `.venv` yorumlayıcısını seçerek çalıştırın:

1. **`01_data_preparation.ipynb`**: Ham PG-19 verisini seçer, temizler, sözlüğü oluşturur ve `data/processed/` dosyalarıyla seçim kaydını yazar.
2. **`02_ngram_models.ipynb`**: Eğitim verisinden sayımları çıkarır; olasılık, smoothing, validation perplexity ve üretim örneklerini hesaplar.
3. **`03_lstm_model.ipynb`**: Token kimliklerini ve 32 token'lık blokları oluşturur, LSTM'yi eğitir, en iyi validation checkpoint'ini kaydeder ve metin üretir.
4. **`04_evaluation.ipynb`**: Sabit model ayarlarıyla test perplexity'sini hesaplar, N-gram modellerini `artifacts/` içine ve sonuçları CSV dosyasına kaydeder.

Üçüncü notebook'un yeni bir eğitim koşusunu başlatmak için kernel'i yeniden başlatıp hücreleri sırayla çalıştırın. Yalnızca epoch sayısını değiştirmek mevcut model ağırlıklarını sıfırlamaz. Yeniden eğitim mevcut `lstm_best.pt` ve eğitim geçmişini değiştireceğinden, korunması gereken eski koşuları önce ayrı bir klasöre kopyalayın.

## Model ayarları

| Model | Ayarlar |
|---|---|
| Bigram | Add-k smoothing, `k=0.01` |
| Trigram | Add-k smoothing, `k=0.001` |
| LSTM | Embedding 128, hidden size 128, 1 katman, tek yönlü |

LSTM eğitimi: sequence length **32**, batch size **32**, Adam öğrenme oranı **0.001**, gradient norm sınırı **1.0**, **10 epoch**. Validation loss'una göre **7. epoch** seçilmiştir. Her kitap başlangıcında `<bos>`, sonunda `<eos>` kullanılır; kitap sınırları birleştirilmez. Kısa blokların dolgu hedefleri `-100` ile kayıp hesabından çıkarılır. Her blokta gizli durum yeniden başlatılır; üretimde gizli durum adımlar arasında taşınır.

## Kayıtlı sonuçlar

| Model | Validation perplexity | Test perplexity |
|---|---:|---:|
| Bigram | 485.25 | 387.81 |
| Trigram | 2845.56 | 2220.55 |
| LSTM, epoch 7 | 200.49 | 171.62 |

Perplexity hesabı noktalama ve `<unk>` dahil gerçek token'ları ve kitap sonu `<eos>` tahminlerini içerir; `<bos>` ve dolgu konumları hedef değildir. Test bölümünde **100.003** tahmin değerlendirilmiştir. Üretimdeki top-k/temperature işlemleri değerlendirmeye uygulanmaz. Bu metrik doğruluk yüzdesi değildir.

Tam test değerleri `results/test_results.csv`, epoch bazındaki ölçümler `results/lstm_history.json` dosyasındadır. Kayıtlı ölçümler GPU koşusuna aittir; cihaz ve sayısal hesaplama farkları küçük değişiklikler oluşturabilir.

## Teslim paketleri

Hazır modeller ve işlenmiş verilerle çalıştırılabilir paket:

```bash
python package_submission.py
```

Çıktı: `dist/nlp_homework_submission.zip`. Arşivi açıp `nlp_homework/` klasörüne girerek uygulamayı çalıştırabilirsiniz. Ham PG-19 kitapları pakete dahil değildir. Uygulamayı çalıştırmak için ham veri gerekmez; veri hazırlamayı yeniden çalıştırmak için gerekir.

Yalnızca kaynak kodu, notebook'lar ve küçük sonuç dosyaları:

```bash
python package_submission.py --source-only
```

Çıktı: `dist/nlp_homework_source.zip`. Bu pakette modeller ve işlenmiş veri bulunmaz. Uygulamayı açmadan önce tam paketteki `artifacts/` klasörünü ekleyin veya ham veriyi indirip notebook'ları sırayla çalıştırın.

GitHub deposunda `artifacts/`, `data/processed/`, `data/pg19/`, `.venv/`, `dist/` ve Python önbellek dosyaları tutulmaz. Modeller ve işlenmiş veriler tam teslim ZIP'inde bulunur.

## Kontroller

```bash
python -m pip check
python -m unittest discover -s tests -v
```

Testler veri yolu seçimini, seçim kaydını, bölüm ayrımını ve ZIP içeriğini kontrol eder. Notebook çıktıları ve Streamlit arayüzü ayrıca çalışma akışının parçasıdır.
