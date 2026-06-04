# 👶 abdimas-wedoro: Dashboard Tumbuh Kembang & Prediksi Gizi Anak

Aplikasi web interaktif berbasis **Streamlit** untuk memantau tumbuh kembang anak, memproyeksikan berat badan (BB) dan tinggi badan (TB) satu bulan ke depan dengan model machine learning **XGBoost Regressor**, serta menentukan status gizi berdasarkan standar referensi **WHO Child Growth Standards (Z-Score)**.

Proyek ini dikembangkan dalam rangka pengabdian masyarakat (Abdimas) di wilayah **Puskesmas Weduro**.

---

## ✨ Fitur Utama
1. **Auto-Load & Auto-Generate Reference**: 
   - Otomatis memuat data historis (`trial_2025.csv`) dan menyusun tabel referensi WHO Z-score (`tabel_who.xlsx`) tanpa perlu unggah manual saat aplikasi pertama kali dibuka.
2. **Pencarian Riwayat Anak Berdasarkan ID**:
   - Fitur pencarian cepat data anak posyandu terdaftar. Mengisi form profil, status gizi terakhir, dan riwayat secara otomatis.
3. **Visualisasi Kurva Pertumbuhan Interaktif (Plotly)**:
   - Menampilkan grafik berat badan menurut umur (BB/U) dan tinggi badan menurut umur (TB/U) lengkap dengan pita standar deviasi WHO (-3 SD, -2 SD, Median, +2 SD) serta titik prediksi target masa depan.
4. **Input Manual Fleksibel**:
   - Mendukung pengujian status gizi secara real-time untuk data anak baru di luar database.
5. **Rekomendasi Nutrisi Dinamis**:
   - Panel tindakan dan intervensi makanan bergizi yang disesuaikan secara otomatis dengan status gizi anak saat ini maupun prediksi bulan depannya (misalnya mitigasi stunting atau obesitas).

---

## 🛠️ Tech Stack & Prasyarat
Untuk menjalankan aplikasi ini secara lokal, pastikan Anda telah menginstal:
* Python 3.8 ke atas
* Streamlit
* Pandas
* Numpy
* XGBoost
* Scikit-Learn
* Openpyxl (untuk membaca & menulis file Excel)
* Plotly (untuk grafik interaktif)

---

## 🚀 Cara Menjalankan Aplikasi

1. **Clone Repositori**:
   ```bash
   git clone https://github.com/Zahidabd12/abdimas-wedoro.git
   cd abdimas-wedoro
   ```

2. **Instalasi Dependensi**:
   ```bash
   pip install streamlit pandas numpy xgboost scikit-learn openpyxl plotly
   ```

3. **Jalankan Aplikasi Streamlit**:
   ```bash
   streamlit run streamlit/app.py
   ```

4. Buka alamat lokal yang tertera di terminal Anda (biasanya `http://localhost:8501`) pada peramban web (browser).

---

## 📂 Struktur Proyek
* `streamlit/app.py`: File program utama aplikasi dashboard Streamlit.
* `trial_2025.csv` / `trial_2026.csv`: Data latih riwayat penimbangan bulanan anak posyandu.
* `tabel_who.xlsx`: File referensi Z-score WHO yang di-generate otomatis berisi parameter LMS (Lambda, Mu, Sigma) pertumbuhan anak.
* `ASP 2025.xlsx` / `ASP 2026.xlsx`: Data master wilayah Puskesmas Weduro.
* `kode.ipynb`: Jupyter notebook analisis dan pelatihan model awal.
