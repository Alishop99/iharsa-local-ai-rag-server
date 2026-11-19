from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import chromadb
import ollama
import time
import json
import os
import uuid
import threading
import traceback
import PyPDF2
import docx

app = Flask(__name__)
CORS(app)

# Hubungkan ke Database Memori IHARSA
client = chromadb.PersistentClient(path="./db_iharsa")
collection = client.get_or_create_collection(name="memori_iharsa")

ALLOWED = ["qwen2.5:7b", "llama3", "gemma4:latest"]
DEFAULT_MODEL = "qwen2.5:7b"
OPTS = {"num_ctx": 8192, "num_predict": 512}
GPU_LOCK = threading.Semaphore(1)
JOBS = {}


def pilih_model(nama):
    return nama if nama in ALLOWED else DEFAULT_MODEL


def cari_konteks(pertanyaan):
    try:
        e = ollama.embeddings(model="nomic-embed-text", prompt=pertanyaan)
        h = collection.query(query_embeddings=[e["embedding"]], n_results=3)
        docs = h["documents"][0] if h["documents"] else []
        metas = h["metadatas"][0] if h["metadatas"] else []
        sumber = [
            f"{m.get('kategori', '-')} (bagian {m.get('bagian', '-')})"
            for m in metas
        ]
        return "\n- ".join(docs), sumber
    except Exception as ex:
        print(f"Peringatan ChromaDB: {ex}")
        return "", []


def bangun_prompt(konteks, pertanyaan):
    if konteks:
        return f"Informasi Internal IHARSA:\n- {konteks}\n\nPertanyaan: {pertanyaan}"
    return pertanyaan


# =====================================================================
# 1. JALUR OLAHPESAN UI ODYSSEUS / OPENAI COMPATIBLE (PORT 3000 / 7011)
# =====================================================================

@app.route('/v1/models', methods=['GET'])
def get_models():
    model_list = []
    for model_name in ALLOWED:
        model_list.append({
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "IHARSA"
        })
    return jsonify({"object": "list", "data": model_list})


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json or {}
    messages = data.get('messages', [])
    stream = data.get('stream', False)

    model_terpilih = data.get('model', DEFAULT_MODEL)
    model_aman = pilih_model(model_terpilih)

    pertanyaan = ""
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            pertanyaan = msg.get('content')
            break

    if not pertanyaan:
        return jsonify({"error": "Tidak ada pertanyaan"}), 400

    # Gunakan RAG otomatis untuk Odysseus UI (fail-safe)
    konteks, _ = cari_konteks(pertanyaan)
    prompt_lengkap = bangun_prompt(konteks, pertanyaan)

    if stream:
        def generate():
            try:
                # Antrean GPU untuk mencegah benturan VRAM (Vulkan ErrorDeviceLost)
                with GPU_LOCK:
                    for chunk in ollama.generate(
                        model=model_aman,
                        prompt=prompt_lengkap,
                        stream=True,
                        options=OPTS
                    ):
                        kata = chunk['response']
                        response_data = {
                            "id": "chatcmpl-iharsa",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_aman,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": kata},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    else:
        try:
            with GPU_LOCK:
                jawaban = ollama.generate(
                    model=model_aman,
                    prompt=prompt_lengkap,
                    options=OPTS
                )
            return jsonify({
                "id": "chatcmpl-iharsa",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_aman,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": jawaban["response"]
                    },
                    "finish_reason": "stop"
                }]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500


# =====================================================================
# 2. JALUR ASINKRON API INTERNAL IHARSA (UNTUK APLIKASI / FRONTEND LAIN)
# =====================================================================

def _worker(job_id, pertanyaan, model, gunakan_memori):
    try:
        if gunakan_memori:
            konteks, sumber = cari_konteks(pertanyaan)
            prompt_lengkap = bangun_prompt(konteks, pertanyaan)
        else:
            sumber = []
            prompt_lengkap = pertanyaan

        with GPU_LOCK:
            r = ollama.generate(
                model=model,
                prompt=prompt_lengkap,
                options=OPTS,
                keep_alive="24h"
            )

        JOBS[job_id] = {
            "status": "selesai",
            "jawaban": r["response"],
            "proses_berpikir": "",
            "sumber": sumber
        }
    except Exception as ex:
        JOBS[job_id] = {"status": "error", "pesan": str(ex)}


@app.route('/tanya_ai_async', methods=['POST'])
def tanya_ai_async():
    data = request.get_json(silent=True) or {}
    pertanyaan = (data.get("pertanyaan") or "").strip()
    gunakan_memori = data.get("gunakan_memori", True)

    if not pertanyaan:
        return jsonify({"status": "error", "pesan": "Pertanyaan kosong"}), 400
    if len(JOBS) > 200:
        for k in list(JOBS.keys())[:100]:
            JOBS.pop(k, None)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "proses"}

    threading.Thread(
        target=_worker,
        args=(job_id, pertanyaan, pilih_model(data.get("model")), gunakan_memori),
        daemon=True
    ).start()
    return jsonify({"status": "sukses", "job_id": job_id})


@app.route('/cek_status/<job_id>')
def cek_status(job_id):
    return jsonify(JOBS.get(job_id, {"status": "error", "pesan": "Job tidak ditemukan"}))


@app.route('/tambah_memori', methods=['POST'])
def tambah_memori():
    try:
        kategori = request.form.get('kategori', 'Umum')
        teks_rahasia = request.form.get('informasi', '')
        file = request.files.get('dokumen')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if ext == 'pdf':
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    teks_rahasia += "\n" + (page.extract_text() or "")
            elif ext in ['doc', 'docx']:
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    teks_rahasia += "\n" + para.text

        if not teks_rahasia.strip():
            return jsonify({"status": "error", "pesan": "Teks kosong!"}), 400

        ukuran_potongan = 1500
        potongan_teks = [
            teks_rahasia[i:i + ukuran_potongan]
            for i in range(0, len(teks_rahasia), ukuran_potongan)
        ]
        ids_vektor, embeddings_vektor, dokumen_vektor, metadatas_vektor = [], [], [], []
        waktu_dasar = int(time.time())

        for index, teks_potong in enumerate(potongan_teks):
            if not teks_potong.strip():
                continue
            embed = ollama.embeddings(model="nomic-embed-text", prompt=teks_potong)
            ids_vektor.append(f"doc_{waktu_dasar}_{index}")
            embeddings_vektor.append(embed["embedding"])
            dokumen_vektor.append(teks_potong)
            metadatas_vektor.append({"kategori": kategori, "bagian": index + 1})

        collection.add(
            ids=ids_vektor,
            embeddings=embeddings_vektor,
            documents=dokumen_vektor,
            metadatas=metadatas_vektor
        )
        return jsonify({
            "status": "sukses",
            "pesan": f"Memori disimpan ({len(dokumen_vektor)} bagian)!"
        })
    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)}), 500


if __name__ == '__main__':
    print("Server IHARSA OpenAI-Compatible & Async Engine menyala di port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
