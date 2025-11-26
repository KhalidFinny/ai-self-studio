# Script untuk verifikasi model yang digunakan
import os
import sys
import tensorflow as tf
from tensorflow import keras

# Tambahkan root ke path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

def check_model_architecture(model_path):
    """Cek arsitektur model dari file .h5"""
    print("=" * 60)
    print("VERIFIKASI MODEL")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan di: {model_path}")
        return False
    
    print(f"[OK] Model ditemukan: {model_path}")
    print("\nMemuat model...")
    
    try:
        model = keras.models.load_model(model_path)
        
        print("\n" + "=" * 60)
        print("ARSITEKTUR MODEL")
        print("=" * 60)
        
        # Cek layer pertama (base model)
        first_layer = model.layers[1]  # Layer 0 biasanya Input, layer 1 adalah base model
        
        print(f"\nBase Model Layer: {first_layer.name}")
        print(f"Layer Type: {type(first_layer).__name__}")
        
        # Cek apakah ResNet50 atau EfficientNet
        layer_name = first_layer.name.lower()
        model_type = first_layer.__class__.__name__
        
        if 'resnet' in layer_name or 'ResNet' in model_type:
            print("\n[OK] MODEL: ResNet50")
            print("   [V] Menggunakan ResNet50 architecture")
        elif 'efficientnet' in layer_name or 'EfficientNet' in model_type:
            print("\n[ERROR] MODEL: EfficientNet")
            print("   [X] Masih menggunakan EfficientNet!")
        else:
            print(f"\n[WARNING] MODEL: {model_type}")
            print("   [?] Tidak bisa menentukan jenis model")
        
        # Tampilkan summary singkat
        print("\n" + "=" * 60)
        print("MODEL SUMMARY (5 layer pertama dan terakhir)")
        print("=" * 60)
        print("\nLayer pertama (Base Model):")
        for i, layer in enumerate(model.layers[:3]):
            print(f"  {i}: {layer.name} - {type(layer).__name__}")
        
        print("\nLayer terakhir:")
        for i, layer in enumerate(model.layers[-3:], start=len(model.layers)-3):
            print(f"  {i}: {layer.name} - {type(layer).__name__}")
        
        # Cek input shape
        print("\n" + "=" * 60)
        print("INPUT/OUTPUT INFO")
        print("=" * 60)
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        print(f"Total Layers: {len(model.layers)}")
        print(f"Total Parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error memuat model: {e}")
        return False

def check_code_files():
    """Cek file kode untuk memastikan menggunakan ResNet50"""
    print("\n" + "=" * 60)
    print("VERIFIKASI KODE")
    print("=" * 60)
    
    # Cek train_efficientnet.py
    train_script = os.path.join(root_dir, "scripts", "train_efficientnet.py")
    if os.path.exists(train_script):
        with open(train_script, 'r') as f:
            content = f.read()
            if 'ResNet50' in content:
                print("[OK] scripts/train_efficientnet.py: Menggunakan ResNet50")
            elif 'EfficientNet' in content:
                print("[ERROR] scripts/train_efficientnet.py: Masih menggunakan EfficientNet!")
            else:
                print("[WARNING] scripts/train_efficientnet.py: Tidak ditemukan referensi model")
    
    # Cek detector
    detector_file = os.path.join(root_dir, "src", "utils", "efficientnet_detector.py")
    if os.path.exists(detector_file):
        with open(detector_file, 'r') as f:
            content = f.read()
            if 'ResNet50GestureDetector' in content:
                print("[OK] src/utils/efficientnet_detector.py: Menggunakan ResNet50GestureDetector")
            elif 'EfficientNetGestureDetector' in content:
                print("[ERROR] src/utils/efficientnet_detector.py: Masih menggunakan EfficientNetGestureDetector!")
            else:
                print("[WARNING] src/utils/efficientnet_detector.py: Tidak ditemukan class detector")
    
    # Cek main.py
    main_file = os.path.join(root_dir, "main.py")
    if os.path.exists(main_file):
        with open(main_file, 'r') as f:
            content = f.read()
            if 'ResNet50GestureDetector' in content and 'resnet50' in content.lower():
                print("[OK] main.py: Menggunakan ResNet50GestureDetector dan path resnet50")
            elif 'EfficientNet' in content or 'efficientnet' in content:
                print("[ERROR] main.py: Masih menggunakan EfficientNet!")
            else:
                print("[WARNING] main.py: Tidak ditemukan referensi model")

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("VERIFIKASI SISTEM: ResNet50 vs EfficientNet")
    print("=" * 60)
    
    # Cek kode
    check_code_files()
    
    # Cek model yang ada
    model_paths = [
        os.path.join(root_dir, "models", "resnet50", "best_model.h5"),
        os.path.join(root_dir, "models", "efficientnet", "best_model.h5")
    ]
    
    print("\n" + "=" * 60)
    print("CEK FILE MODEL")
    print("=" * 60)
    
    resnet_exists = os.path.exists(model_paths[0])
    efficientnet_exists = os.path.exists(model_paths[1])
    
    if resnet_exists:
        print(f"[OK] Model ResNet50 ditemukan: {model_paths[0]}")
        check_model_architecture(model_paths[0])
    else:
        print(f"[WARNING] Model ResNet50 tidak ditemukan: {model_paths[0]}")
        print("   -> Jalankan training untuk membuat model ResNet50")
    
    if efficientnet_exists:
        print(f"\n[WARNING] Model EfficientNet masih ada: {model_paths[1]}")
        print("   -> File ini bisa dihapus jika sudah tidak digunakan")
    else:
        print(f"\n[OK] Model EfficientNet tidak ditemukan (sudah diganti)")
    
    print("\n" + "=" * 60)
    print("KESIMPULAN")
    print("=" * 60)
    
    if resnet_exists and not efficientnet_exists:
        print("[OK] Sistem sudah menggunakan ResNet50")
        print("   - Model ResNet50 tersedia")
        print("   - Model EfficientNet sudah tidak digunakan")
    elif resnet_exists and efficientnet_exists:
        print("[WARNING] Kedua model ada")
        print("   - Pastikan main.py menggunakan model ResNet50")
    else:
        print("[ERROR] Model ResNet50 belum dibuat")
        print("   - Jalankan: python scripts/train_efficientnet.py")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

