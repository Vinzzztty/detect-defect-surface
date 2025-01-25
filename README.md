# Deteksi Kerusakan Permukaan Kain

Aplikasi ini dirancang untuk mendeteksi kerusakan pada permukaan kain menggunakan model deep learning berbasis **EfficientNet-B0**. Aplikasi ini membantu mempermudah inspeksi kualitas kain secara otomatis dengan memprediksi 4 jenis kerusakan:

-   **Hole** (lubang)
-   **Line** (garis)
-   **Stain** (noda)
-   **Thread** (benang)

## Fitur

-   **Antarmuka Pengguna Interaktif:** Dibangun menggunakan framework **Streamlit**.
-   **Model Prediksi Akurat:** Menggunakan model **EfficientNet-B0** yang telah dilatih.
-   **Hasil Probabilitas:** Menampilkan probabilitas untuk setiap kelas kerusakan.

---

## Instalasi Proyek

Aplikasi ini membutuhkan **Python 3.11**. Ikuti langkah-langkah berikut untuk menjalankan proyek:

### 1. Clone Repository

Clone repository ke komputer lokal Anda:

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Buat Virtual Environment

Buat dan aktifkan virtual environment untuk proyek:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Untuk macOS/Linux
.venv\Scripts\activate    # Untuk Windows
```

### 3. Install Dependensi

Install semua dependensi yang dibutuhkan menggunakan `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

Jalankan aplikasi **Streamlit** menggunakan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada alamat default `http://localhost:8501`.

---

## Struktur File

Berikut adalah struktur file utama proyek:

```
project-folder/
│
├── .venv/                # Virtual environment
├── models/               # Model terlatih
│   └── Best Model EfficientNet B0.pth
├── notebooks/            # Notebook Jupyter untuk eksperimen
├── app.py                # File utama aplikasi Streamlit
├── README.md             # Dokumentasi proyek
├── requirements.txt      # Daftar dependensi Python
└── .gitignore            # File untuk mengabaikan file/direktori tertentu
```

---

## Catatan Tambahan

1. **Model Path:** Pastikan file model terlatih berada di folder `models/` dengan nama `Best Model EfficientNet B0.pth`.
2. **Python Version:** Gunakan Python versi 3.11 untuk kompatibilitas penuh.
3. **Dependensi:** Jika ada kesalahan instalasi, pastikan versi pustaka yang digunakan sesuai dengan `requirements.txt`.

---
