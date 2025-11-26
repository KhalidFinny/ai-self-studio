# Setup Python Interpreter di IDE

Jika IDE masih tidak mengenali library TensorFlow, ikuti langkah berikut:

## VS Code / Cursor

### 1. Pilih Python Interpreter

1. Tekan `Ctrl+Shift+P` (atau `Cmd+Shift+P` di Mac)
2. Ketik: `Python: Select Interpreter`
3. Pilih: `.venv\Scripts\python.exe` (atau `.\venv\Scripts\python.exe`)

### 2. Reload Window

1. Tekan `Ctrl+Shift+P`
2. Ketik: `Developer: Reload Window`
3. Pilih untuk reload

### 3. Verifikasi

Buka terminal di VS Code dan jalankan:
```bash
python -c "from tensorflow.keras.applications import ResNet50; print('OK')"
```

Jika berhasil, berarti interpreter sudah benar.

## PyCharm

1. File → Settings → Project → Python Interpreter
2. Klik gear icon → Add
3. Pilih Existing Environment
4. Pilih: `.venv\Scripts\python.exe`
5. Apply dan OK

## Troubleshooting

### Masih tidak dikenali?

1. **Restart IDE** sepenuhnya
2. **Cek virtual environment aktif**:
   ```bash
   .venv\Scripts\activate
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

3. **Install ulang jika perlu**:
   ```bash
   .venv\Scripts\python.exe -m pip install tensorflow
   ```

4. **Cek file `.vscode/settings.json`** sudah benar:
   ```json
   {
       "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
   }
   ```

## Catatan

- File `.vscode/settings.json` sudah diupdate untuk menggunakan `.venv`
- Pastikan virtual environment `.venv` aktif saat menjalankan script
- Library TensorFlow sudah terinstall di `.venv`


