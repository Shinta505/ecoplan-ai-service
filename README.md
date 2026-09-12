# ♻️ EcoPlan AI Microservice (ResNet50 & ONNX)

> **Backend AI Service untuk Klasifikasi Sampah Otomatis berbasis Deep Learning (Transfer Learning ResNet50), ONNX Runtime, dan FastAPI.**

Repositori ini merupakan *microservice* kecerdasan buatan terpisah yang dikembangkan sebagai bagian dari sistem aplikasi pengelolaan sampah **EcoPlan**. Layanan ini bertugas menerima input citra sampah dari *frontend* (Next.js) melalui perantara *backend* utama (Golang), memprosesnya menggunakan model *Convolutional Neural Network* (CNN) berarsitektur **ResNet50**, dan mengembalikan hasil klasifikasi serta pengelompokan jenis sampah secara *real-time*.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10.x
* **Framework:** FastAPI & Uvicorn
* **Model Format:** ONNX (Open Neural Network Exchange) via `onnxruntime`
* **Image Processing:** Pillow (PIL) & NumPy
* **Deployment Ready:** Docker & Uvicorn ASGI Server

---

## 📂 Struktur Direktori Proyek

```text
ecoplan-ai-service/
│
├── .venv/                              # Virtual Environment Python
├── .gitignore                          # Pengaturan file yang diabaikan Git
├── Dockerfile                          # Konfigurasi container untuk deployment cloud
├── requirements.txt                    # Daftar pustaka dependencies
├── main.py                             # Kode utama server FastAPI & ONNX inference
└── resnet50_garbage_ecoplan.onnx       # File bobot model AI hasil konversi ONNX

```

---

## ⚙️ Cara Instalasi & Menjalankan di Lokal

Ikuti langkah-langkah berikut untuk menjalankan *microservice* ini di komputer lokalmu:

### 1. Clone Repositori & Masuk ke Folder

```bash
git clone https://github.com/Shinta505/ecoplan-ai-service.git
cd ecoplan-ai-service

```

### 2. Buat dan Aktifkan Virtual Environment

* **Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate

```


* **Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate

```



### 3. Install Dependencies

Pastikan *pip* berada di versi terbaru, lalu instal seluruh pustaka yang diperlukan:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

```

### 4. Jalankan Server FastAPI

Pastikan file `resnet50_garbage_ecoplan.onnx` sudah berada di dalam folder yang sama dengan `main.py`, lalu jalankan perintah:

```bash
uvicorn main:app --reload

```

Server lokal akan aktif di `http://127.0.0.1:8000`. Kamu bisa membuka `http://127.0.0.1:8000/docs` di browser untuk mengakses dokumentasi interaktif Swagger UI.

---

## 🔌 Dokumentasi Endpoints API

### 1. Root Check (Health Check)

* **URL:** `GET /`
* **Deskripsi:** Memeriksa apakah layanan AI aktif dan berjalan normal.
* **Response Contoh:**
```json
{
  "status": "ok",
  "message": "EcoPlan AI Microservice is running!"
}

```



### 2. Prediksi Kategori Sampah

* **URL:** `POST /api/v1/detections`
* **Deskripsi:** Mengunggah gambar sampah untuk diklasifikasikan oleh model ONNX ResNet50.
* **Request Body:** `multipart/form-data` dengan key `image` (berformat file `.jpg`, `.jpeg`, atau `.png`).
* **Response Contoh:**
```json
{
  "success": true,
  "data": {
    "predicted_category": "battery",
    "waste_group": "Residu (B3/Sulit Didaur Ulang)",
    "confidence_score": 100.0,
    "model_used": "ResNet50_ONNX"
  }
}

```



---

## 🐳 Panduan Deployment (Docker)

Jika kamu ingin mendeploy layanan ini ke server *cloud* (seperti Render, Railway, atau VPS), kamu bisa menggunakan Docker dengan perintah berikut:

1. **Build Docker Image:**
```bash
docker build -t ecoplan-ai-service .

```


2. **Run Docker Container:**
```bash
docker run -d -p 8000:8000 ecoplan-ai-service

```



---

## 📜 Lisensi & Hak Cipta

Proyek ini dikembangkan untuk
