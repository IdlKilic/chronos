import sys
import os
import torch
from PyQt5.QtWidgets import QApplication
from simple_webcam_gui import WebcamApp

def main():
    # PyQt uygulamasını başlat
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern görünüm
    
    # Başlangıç kontrolleri
    check_requirements()
    
    # Ana pencereyi oluştur ve göster
    window = WebcamApp()
    window.show()
    
    # Uygulamayı çalıştır
    sys.exit(app.exec_())

def check_requirements():
    """Gerekli kütüphaneleri ve dosyaları kontrol eder."""
    # PyTorch GPU kontrol
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print(f"CUDA kullanılabilir: {cuda_available}")
        print(f"Cihaz sayısı: {device_count}")
        print(f"Cihaz adı: {device_name}")
    else:
        print("CUDA kullanılamıyor. CPU üzerinde çalışılacak.")
    
    # Veri klasörü kontrolü
    cleandataset_dir = "cleandataset"
    if not os.path.exists(cleandataset_dir):
        print(f"Uyarı: '{cleandataset_dir}' klasörü bulunamadı.")
    else:
        # Alt klasörleri kontrol et (001, 002, ...)
        subfolders = [f for f in os.listdir(cleandataset_dir) 
                       if os.path.isdir(os.path.join(cleandataset_dir, f))]
        print(f"'{cleandataset_dir}' klasöründe {len(subfolders)} alt klasör bulundu.")
    
    # Model kontrol
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_files = [f for f in os.listdir(model_dir) 
                    if f.endswith(('.pth', '.pt')) and os.path.isfile(os.path.join(model_dir, f))]
    
    if len(model_files) == 0:
        print(f"Uyarı: '{model_dir}' klasöründe model dosyası bulunamadı.")
    else:
        print(f"'{model_dir}' klasöründe {len(model_files)} model dosyası bulundu.")
    
    # Sonuçlar klasörü
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

if __name__ == "__main__":
    main()