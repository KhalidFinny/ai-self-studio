# Fist Trigger Photo Booth

Aplikasi Photo Booth yang menggunakan deteksi gesture (fist) untuk memicu foto secara otomatis melalui Sony Imaging Edge Remote.

## Struktur Proyek

```
self(self)Studio/
├── src/                    # Source code utama
│   ├── utils/             # Utility modules
│   │   ├── camera.py      # Handler untuk webcam
│   │   ├── detection.py   # Deteksi gesture menggunakan YOLO
│   │   └── imaging_edge.py # Kontrol Sony Imaging Edge Remote
│   └── gui/               # GUI components
│       └── app.py         # Aplikasi GUI utama
├── scripts/               # Script utility
│   ├── test_camera.py            # Script untuk test koneksi webcam
│   ├── preprocess_dataset.py     # Preprocessing dataset untuk training
│   ├── train_efficientnet.py     # Training ResNet50
│   └── run_training.py           # Pipeline training (preprocessing + training)
├── dataset/              # Dataset asli (palm/, fist/)
├── dataset_processed/    # Dataset setelah preprocessing (train/, val/, test/)
├── models/               # Model weights
│   └── resnet50/         # Model ResNet50 yang sudah di-train
├── runs/                 # Hasil training YOLO
├── main.py              # Entry point aplikasi
└── requirements.txt     # Dependencies
```

## Instalasi

1. Buat virtual environment (opsional tapi disarankan):
```bash
python -m venv selfphoto-env
```

2. Aktifkan virtual environment:
```bash
# Windows
selfphoto-env\Scripts\activate

# Linux/Mac
source selfphoto-env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pastikan model YOLO sudah ada di `runs/detect/train/weights/best.pt`

## Penggunaan

### Menjalankan Aplikasi Utama

```bash
python main.py
```

### Test Koneksi Webcam

```bash
python scripts/test_camera.py
```

## Training Model ResNet50

Proyek ini juga menyediakan script untuk training model ResNet50 untuk klasifikasi palm dan fist.

### Langkah-langkah Training

1. **Preprocessing Dataset** (Resize, Augmentasi, Split):
```bash
python scripts/preprocess_dataset.py
```

2. **Training Model**:
```bash
python scripts/train_efficientnet.py
```

Atau jalankan keduanya sekaligus:
```bash
python scripts/run_training.py
```

### Detail Training

- **Model**: ResNet50 (pretrained ImageNet)
- **Input Size**: 224x224
- **Classes**: 2 (palm, fist)
- **Augmentasi**: Brightness/contrast, rotation, zoom, blur, horizontal flip
- **Split Dataset**: Train 70%, Validation 20%, Test 10%
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Binary Crossentropy (sesuai permintaan)
- **Output**: 1 neuron dengan sigmoid activation (0=palm, 1=fist)
- **Target Accuracy**: > 95%

Model yang sudah di-train akan disimpan di `models/resnet50/`.

Lihat `scripts/README.md` untuk informasi lebih detail tentang training.

## Cara Kerja

1. Aplikasi akan membuka webcam dan menampilkan feed video
2. Deteksi gesture "fist" menggunakan model YOLO
3. Jika fist terdeteksi secara konsisten selama 5 frame, akan memicu countdown
4. Setelah countdown selesai, aplikasi akan:
   - Mencari window Sony Imaging Edge Remote
   - Fokus ke window tersebut
   - Mengklik tombol shutter untuk mengambil foto

## Kontrol

- **Esc**: Keluar dari aplikasi (exit fullscreen)
- **Fist Gesture**: Tunjukkan kepalan tangan untuk memicu foto

## Konfigurasi

Anda dapat mengubah konfigurasi di `main.py`:

- `camera_id`: ID camera (default: 0)
- `width`, `height`: Resolusi camera (default: 1280x720)
- `model_path`: Path ke model YOLO
- `confidence`: Threshold confidence untuk deteksi (default: 0.85)
- `countdown_seconds`: Durasi countdown (default: 3)

## Requirements

- Python 3.10+
- OpenCV
- Ultralytics YOLO
- TensorFlow (untuk training ResNet50)
- Tkinter
- PyAutoGUI
- PyGetWindow
- PIL/Pillow
- scikit-learn
- matplotlib
- seaborn
- tqdm

Lihat `requirements.txt` untuk daftar lengkap dependencies.

