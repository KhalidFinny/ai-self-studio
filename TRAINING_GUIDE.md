# Panduan Training ResNet50

Panduan lengkap untuk training model ResNet50 untuk klasifikasi palm dan fist.

## Prerequisites

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Struktur Dataset**:
Pastikan dataset sudah ada di folder `dataset/` dengan struktur:
```
dataset/
├── palm/    # Gambar palm (1861 gambar)
└── fist/    # Gambar fist (1735 gambar)
```

## Langkah-langkah Training

### Option 1: Jalankan Pipeline Lengkap (Recommended)

Jalankan preprocessing dan training sekaligus:
```bash
python scripts/run_training.py
```

### Option 2: Jalankan Manual

#### Step 1: Preprocessing Dataset

Preprocessing akan:
- Resize semua gambar ke 224x224
- Melakukan augmentasi pada training set
- Split dataset: 70% train, 20% val, 10% test

```bash
python scripts/preprocess_dataset.py
```

**Output**: Folder `dataset_processed/` dengan struktur:
```
dataset_processed/
├── train/
│   ├── palm/    # Gambar palm untuk training (dengan augmentasi)
│   └── fist/    # Gambar fist untuk training (dengan augmentasi)
├── val/
│   ├── palm/    # Gambar palm untuk validation
│   └── fist/    # Gambar fist untuk validation
└── test/
    ├── palm/    # Gambar palm untuk test
    └── fist/    # Gambar fist untuk test
```

#### Step 2: Training Model

Training ResNet50:
```bash
python scripts/train_efficientnet.py
```

**Output**: Folder `models/resnet50/` berisi:
- `best_model.h5` - Model terbaik
- `final_model.h5` - Model final
- `fine_tuned_model.h5` - Model setelah fine-tuning (jika dilakukan)
- `training_history.png` - Grafik training
- `confusion_matrix.png` - Confusion matrix
- `classification_report.txt` - Laporan klasifikasi

## Konfigurasi Training

### Hyperparameters Default

- **Batch Size**: 32
- **Epochs**: 50 (dengan early stopping)
- **Learning Rate**: 0.0001
- **Input Size**: 224x224
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy (sesuai permintaan)
- **Output**: 1 neuron dengan sigmoid (0=palm, 1=fist)

### Callbacks

- **ModelCheckpoint**: Simpan model terbaik berdasarkan validation accuracy
- **EarlyStopping**: Stop training jika tidak ada improvement (patience=10)
- **ReduceLROnPlateau**: Reduce learning rate jika loss tidak turun (patience=5)

### Fine-tuning

Jika accuracy masih di bawah 95%, model akan secara otomatis melakukan fine-tuning:
- Unfreeze top layers dari base model
- Learning rate dikurangi menjadi 0.1x
- Training dilanjutkan hingga 20 epoch

## Monitoring Training

Training akan menampilkan:
- Progress bar untuk setiap epoch
- Training dan validation accuracy/loss
- Early stopping jika diperlukan
- Learning rate adjustment

Setelah training selesai:
- Test accuracy akan ditampilkan
- Confusion matrix akan di-generate
- Classification report akan disimpan

## Troubleshooting

### Error: "Dataset tidak ditemukan"
- Pastikan folder `dataset_processed/` sudah ada (jalankan preprocessing terlebih dahulu)

### Error: "Out of memory"
- Kurangi `batch_size` di script training (default: 32, coba 16 atau 8)
- Pastikan GPU memiliki cukup memory (jika menggunakan GPU)

### Accuracy rendah (< 90%)
- Cek apakah augmentasi sudah dilakukan dengan benar
- Pastikan dataset seimbang antara palm dan fist
- Coba fine-tuning manual dengan mengubah learning rate

### Training terlalu lama
- Gunakan GPU jika tersedia (TensorFlow akan otomatis detect)
- Kurangi jumlah epoch atau gunakan early stopping
- Kurangi augmentasi jika tidak diperlukan

## Hasil yang Diharapkan

Target training:
- **Validation Accuracy**: > 95%
- **Test Accuracy**: > 95%
- **Tidak Overfitting**: Training accuracy ≈ Validation accuracy

Jika target tercapai, model siap digunakan untuk inference!

## Menggunakan Model yang Sudah Di-train

Model yang sudah di-train dapat digunakan untuk inference dengan TensorFlow:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('models/resnet50/best_model.h5')

# Preprocess image
img = cv2.imread('path/to/image.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)
class_idx = np.argmax(predictions[0])
classes = ['palm', 'fist']
print(f"Predicted: {classes[class_idx]} ({predictions[0][class_idx]*100:.2f}%)")
```

## Next Steps

Setelah model berhasil di-train:
1. Integrate model ke aplikasi utama (update `src/utils/detection.py`)
2. Test model dengan gambar baru
3. Fine-tune jika diperlukan
4. Deploy model untuk production

