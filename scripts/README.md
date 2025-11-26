# Scripts Training

Script-script untuk preprocessing dataset dan training model EfficientNet-B0.

## Preprocessing Dataset

Script untuk melakukan preprocessing dataset sebelum training:

```bash
python scripts/preprocess_dataset.py
```

Script ini akan:
- Resize semua gambar ke 224x224 (ukuran input EfficientNet-B0)
- Melakukan augmentasi pada training set:
  - Random brightness/contrast
  - Random rotation
  - Random zoom
  - Random background blur
  - Horizontal flip
- Split dataset menjadi:
  - Train: 70%
  - Validation: 20%
  - Test: 10%

Output akan disimpan di folder `dataset_processed/`.

## Training EfficientNet-B0

Script untuk training model EfficientNet-B0:

```bash
python scripts/train_efficientnet.py
```

**Catatan**: Pastikan sudah menjalankan preprocessing terlebih dahulu!

Script ini akan:
- Load EfficientNet-B0 pretrained (ImageNet)
- Freeze base model untuk transfer learning
- Training dengan:
  - Optimizer: Adam (lr=0.0001)
  - Loss: Binary Crossentropy (sesuai permintaan)
  - Output: 1 neuron dengan sigmoid activation (0=palm, 1=fist)
  - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Fine-tuning jika accuracy < 95%
- Evaluate model pada test set
- Simpan model dan hasil evaluasi

Model yang sudah di-train akan disimpan di folder `models/efficientnet/`.

## Hasil Training

Setelah training selesai, akan tersedia:
- `best_model.h5` - Model terbaik berdasarkan validation accuracy
- `final_model.h5` - Model final setelah training
- `fine_tuned_model.h5` - Model setelah fine-tuning (jika dilakukan)
- `training_history.png` - Grafik training history
- `confusion_matrix.png` - Confusion matrix pada test set
- `classification_report.txt` - Laporan klasifikasi detail

