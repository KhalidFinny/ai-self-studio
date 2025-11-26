# Setup IDE untuk Proyek

Panduan untuk mengkonfigurasi IDE (VS Code/Cursor) agar mengenali semua library yang diperlukan.

## Langkah-langkah

### 1. Pastikan Virtual Environment Aktif

Library-library sudah terinstall di virtual environment `selfphoto-env`. Pastikan IDE menggunakan interpreter dari virtual environment ini.

### 2. Konfigurasi VS Code/Cursor

File `.vscode/settings.json` sudah dibuat untuk mengkonfigurasi Python interpreter. Jika IDE masih tidak mengenali library:

1. **Reload Window**:
   - Tekan `Ctrl+Shift+P` (atau `Cmd+Shift+P` di Mac)
   - Ketik "Reload Window"
   - Pilih "Developer: Reload Window"

2. **Pilih Python Interpreter Manual**:
   - Tekan `Ctrl+Shift+P`
   - Ketik "Python: Select Interpreter"
   - Pilih: `.\selfphoto-env\Scripts\python.exe`

### 3. Verifikasi Instalasi

Jalankan di terminal untuk verifikasi:
```bash
selfphoto-env\Scripts\python.exe -c "import tensorflow; import sklearn; import seaborn; import tqdm; print('OK')"
```

### 4. Jika Masih Ada Error

Jika library masih tidak dikenali setelah reload:

1. **Restart IDE** sepenuhnya
2. **Cek Python Extension** di VS Code/Cursor sudah terinstall
3. **Pastikan virtual environment aktif** di terminal:
   ```bash
   selfphoto-env\Scripts\activate
   ```

## Library yang Sudah Terinstall

- ✅ TensorFlow 2.20.0
- ✅ scikit-learn 1.7.2
- ✅ matplotlib 3.10.3
- ✅ seaborn 0.13.2
- ✅ tqdm 4.67.1
- ✅ numpy 1.26.4
- ✅ OpenCV
- ✅ Dan lainnya...

## Catatan

Ada warning tentang protobuf version conflict dengan mediapipe, tapi ini tidak akan menghalangi penggunaan. Mediapipe biasanya masih bisa bekerja dengan protobuf yang lebih baru.

## Troubleshooting

### Error: "Import could not be resolved"

1. Pastikan virtual environment dipilih sebagai Python interpreter
2. Reload window di IDE
3. Restart IDE jika perlu

### Error: "Module not found"

1. Pastikan virtual environment aktif
2. Install ulang dengan:
   ```bash
   selfphoto-env\Scripts\python.exe -m pip install -r requirements.txt
   ```

### IDE masih tidak mengenali library

1. Tutup dan buka kembali project
2. Pastikan `.vscode/settings.json` ada dan benar
3. Cek di terminal apakah library bisa diimport:
   ```bash
   selfphoto-env\Scripts\python.exe -c "import tensorflow"
   ```

