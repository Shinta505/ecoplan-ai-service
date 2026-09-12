import io
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort

app = FastAPI(
    title="EcoPlan AI Microservice",
    description="Microservice untuk deteksi sampah menggunakan model ResNet50 ONNX",
    version="1.0.0"
)

# Load model ONNX (Dilakukan sekali saat server menyala agar inferensi cepat)
# Pastikan file .onnx berada di folder yang sama dengan main.py
MODEL_PATH = "resnet50_garbage_ecoplan.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"[INFO] Model ONNX berhasil dimuat. Input node: {input_name}")
except Exception as e:
    print(f"[ERROR] Gagal memuat model ONNX: {e}")
    print("Pastikan file 'resnet50_garbage_ecoplan.onnx' sudah dipindahkan ke direktori ini.")

# Berdasarkan dataset sumn2u/garbage-classification-v2 (diurutkan alfabetis otomatis oleh ImageDataGenerator saat training)
CLASS_LABELS = [
    "battery", "biological", "cardboard", 
    "clothes", "glass", "metal", "paper", 
    "plastic", "shoes", "trash"
]

# Fungsi pembantu untuk mengkategorikan ke kelompok sampah (Sesuai Blueprint EcoPlan)
def get_waste_group(category_name: str) -> str:
    organik = ["biological"]
    residu = ["trash", "battery"] # Baterai bisa masuk B3/Residu tergantung klasifikasi bank sampah
    # Sisanya masuk Non-Organik (bisa didaur ulang)
    if category_name in organik:
        return "Organik"
    elif category_name in residu:
        return "Residu (B3/Sulit Didaur Ulang)"
    else:
        return "Non-Organik (Dapat Didaur Ulang)"

def preprocess_image_resnet50(image_bytes: bytes) -> np.ndarray:
    """
    Mereplikasi logika tf.keras.applications.resnet50.preprocess_input
    tanpa menggunakan library TensorFlow agar server tetap ringan.
    """
    try:
        # Buka gambar menggunakan Pillow
        img = Image.open(io.BytesIO(image_bytes))
        
        # Konversi ke RGB jika gambar memiliki Alpha channel (RGBA) atau Grayscale
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Resize ke 224x224 (Target dimensi ResNet50)
        img = img.resize((224, 224))
        
        # Konversi ke Numpy Array float32
        x = np.array(img, dtype=np.float32)
        
        # Logika Preprocessing Keras ResNet50 (Mode 'caffe'):
        # 1. Ubah format dari RGB ke BGR
        x = x[..., ::-1]
        
        # 2. Zero-center menggunakan rata-rata pixel dataset ImageNet
        mean_imagenet = [103.939, 116.779, 123.68]
        x[..., 0] -= mean_imagenet[0]
        x[..., 1] -= mean_imagenet[1]
        x[..., 2] -= mean_imagenet[2]
        
        # Tambahkan dimensi batch (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)
        return x
    
    except Exception as e:
        raise ValueError(f"Gagal memproses gambar: {str(e)}")

@app.post("/api/v1/detections")
async def detect_waste(image: UploadFile = File(...)):
    """
    Endpoint untuk menerima gambar dari Backend Golang / Frontend Next.js
    dan mengembalikan hasil klasifikasi AI.
    """
    # 1. Validasi Ekstensi File
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPEG/PNG).")
    
    try:
        # 2. Baca file bytes
        contents = await image.read()
        
        # 3. Preprocessing (Resize & Normalisasi ResNet50)
        input_data = preprocess_image_resnet50(contents)
        
        # 4. Jalankan Inferensi menggunakan ONNX
        outputs = session.run(None, {input_name: input_data})
        
        # Output dari ResNet50 (dengan softmax) adalah probabilitas array shape (1, 12)
        probabilities = outputs[0][0]
        
        # 5. Cari index dengan nilai probabilitas tertinggi
        predicted_index = int(np.argmax(probabilities))
        confidence_score = float(probabilities[predicted_index])
        
        # 6. Mapping hasil ke format JSON response
        predicted_category = CLASS_LABELS[predicted_index]
        waste_group = get_waste_group(predicted_category)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "predicted_category": predicted_category,
                "waste_group": waste_group,
                "confidence_score": round(confidence_score * 100, 2), # Konversi ke persen (misal: 98.45)
                "model_used": "ResNet50_ONNX"
            }
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server AI: {str(e)}")

@app.get("/")
def health_check():
    """Endpoint untuk mengecek apakah server AI hidup"""
    return {"status": "ok", "message": "EcoPlan AI Microservice is running!"}

if __name__ == "__main__":
    # Menjalankan server FastAPI di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)