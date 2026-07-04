# Klasifikasi Motif Batik Nitik Menggunakan ShuffleNetV2 Berbasis Transfer Learning

Aplikasi web berbasis Streamlit untuk mengklasifikasikan motif Batik Nitik menggunakan model deep learning ShuffleNetV2. Pengguna dapat mengunggah citra batik dan memperoleh hasil klasifikasi motif beserta tingkat keyakinan (confidence) secara langsung.

Coba aplikasinya:

## Tentang

Batik Nitik merupakan salah satu motif batik khas Yogyakarta. Proyek ini merupakan bagian dari penelitian skripsi yang mengimplementasikan arsitektur ShuffleNetV2 dengan pendekatan transfer learning untuk klasifikasi citra motif Batik Nitik. Model dilatih menggunakan dataset Batik Nitik 960 yang terdiri atas 960 citra pada 60 kelas motif.

## Fitur

- Klasifikasi motif batik dari citra yang diunggah, mendukung unggah banyak file sekaligus
- Menampilkan tingkat keyakinan (confidence) beserta top-5 probabilitas prediksi
- Indikator visual (hijau/kuning/merah) berdasarkan tingkat keyakinan prediksi
- Unduh hasil prediksi dalam format teks

## Model
 
| Keterangan | Detail |
|---|---|
| Arsitektur | ShuffleNetV2 x1.0 |
| Dataset Pelatihan | Batik Nitik 960 (60 kelas) |
| Jumlah Parameter | 1.3151 |
| Ukuran Model | 5.3MB |
 
## Instalasi Lokal
 
Klon repositori ini:
 
```bash
git clone https://github.com/username/batik-nitik.git
cd batik-nitik
```
 
Instal dependensi:
 
```bash
pip install -r requirements.txt
```
 
Jalankan aplikasi:
 
```bash
streamlit run app.py
```
 
Aplikasi akan terbuka otomatis di `http://localhost:8501`.
