# Dataset Riset: Evaluasi dan Implementasi Arsitektur Server AI Lokal (IHARSA) Berbasis LLM dan RAG

## Deskripsi Singkat (Overview)
Dataset ini berisi skrip implementasi, konfigurasi lingkungan, dan log pemecahan masalah (*troubleshooting logs*) dari pembangunan ekosistem *Large Language Model* (LLM) lokal mandiri (**IHARSA**) yang terintegrasi dengan *Retrieval-Augmented Generation* (RAG). Sistem ini dirancang untuk beroperasi secara stabil di atas perangkat keras kelas konsumen dengan keterbatasan VRAM (**AMD Radeon RX 580 8GB** menggunakan *Vulkan backend*).

---

## Daftar Berkas dalam Dataset (File Manifest)
1. **`README.md`**: Dokumentasi arsitektur sistem, temuan penelitian, dan panduan reproduksi.
2. **`api_ai.py`**: Skrip *middleware* API Python (Flask) final yang kompatibel dengan standar OpenAI (`/v1/models`, `/v1/chat/completions`), mendukung *streaming* SSE, antrean sinkronisasi GPU (`threading.Semaphore`), serta *endpoint* asinkron dan injeksi dokumen PDF/DOCX ke ChromaDB.
3. **`troubleshooting_logs.txt`**: Rekaman log terminal asli dari PM2 (`api-iharsa-out.log` dan `api-iharsa-error.log`) yang mendokumentasikan kegagalan HTTP 404, HTTP 500 (*Flask TypeError*), dan galat Vulkan (`vk::Queue::submit: ErrorDeviceLost`), hingga status stabil akhir.
4. **`docker-compose.yml`**: Referensi konfigurasi Docker untuk membuka akses jaringan lokal port `7011:7000` pada antarmuka Odysseus UI.

---

## Spesifikasi Lingkungan Sistem (System Environment)
- **Sistem Operasi**: Ubuntu Linux 26.04
- **Perangkat Keras (GPU)**: AMD Radeon RX 580 8GB (Vulkan / ROCm backend)
- **Mesin Inferensi LLM**: Ollama (Port bawaan `11434`)
- **Middleware / API Bridge**: Python 3 (Flask, Flask-CORS), dikelola dengan PM2 (`api_iharsa`, Port `5000`)
- **Antarmuka Pengguna (UI)**: Odysseus UI / OpenClaw (Port `3000` / `7011`)
- **Basis Data Vektor**: ChromaDB (`./db_iharsa`, koleksi `memori_iharsa`)
- **Model LLM**: `qwen2.5:7b` (default), `llama3`, `gemma4:latest`, dan `bonsai27b` (uji beban)
- **Model Embedding**: `nomic-embed-text`

---

## Ringkasan Temuan Teknis (Key Findings)
1. **Context Window Overflow & Timeout**: Penumpukan riwayat obrolan dan beban model berukuran besar (`bonsai27b`) melampaui kapasitas konteks/VRAM dan menyebabkan *timeout* layanan Ollama.
2. **Ketidaksinkronan Rute API (HTTP 404)**: Klien berbasis OpenAI-compatible memerlukan definisi rute eksplisit `/v1/models` dan `/v1/chat/completions` pada *middleware* Flask.
3. **Konflik Memori GPU / VRAM Exhaustion (Vulkan `ErrorDeviceLost`)**: Eksekusi serentak antara proses *embedding* ChromaDB dan inferensi LLM pada GPU 8GB memicu galat `vk::Queue::submit: ErrorDeviceLost (status code: 500)`.
4. **Mitigasi Sistemik**: Penerapan penguncian antrean `GPU_LOCK = threading.Semaphore(1)`, pembatasan parameter (`num_ctx: 8192`, `num_predict: 512`), blok *try-except fail-safe* pada fungsi pencarian vektor (`cari_konteks`), serta pemindahan *default* ke model kelas 7B/8B berhasil menghilangkan galat HTTP 500 dan menghasilkan *streaming* respons yang stabil.

---

## Panduan Reproduksi (Reproducibility Steps)
1. **Siapkan Lingkungan Python & Dependensi**:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install flask flask-cors chromadb ollama PyPDF2 python-docx werkzeug
   ```
2. **Pastikan Layanan Ollama & Model Tersedia**:
   ```bash
   sudo systemctl start ollama
   ollama pull qwen2.5:7b
   ollama pull llama3
   ollama pull gemma4:latest
   ollama pull nomic-embed-text
   ```
3. **Jalankan Middleware API dengan PM2**:
   ```bash
   pm2 start api_ai.py --name api_iharsa --interpreter ./env/bin/python
   pm2 save
   pm2 startup
   ```
4. **Hubungkan Antarmuka Odysseus UI**:
   - Buka pengaturan API (*Settings -> API Connections*).
   - Masukkan Base URL: `http://<IP-SERVER>:5000/v1` (misal `http://192.168.1.18:5000/v1`).
   - Pilih model yang tersedia (`qwen2.5:7b`, `llama3`, atau `gemma4:latest`).
