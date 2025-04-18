import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
                             QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class WebcamApp(QMainWindow):
    def __init__(self):
        super().__init__() 
        
        # Ana pencere ayarları
        self.setWindowTitle("Webcam Yüz Yaşlandırma")
        self.setGeometry(100, 100, 1000, 600)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Video görüntüleme bölgesi
        self.video_layout = QHBoxLayout()
        
        # Orijinal webcam görüntüsü
        self.webcam_label = QLabel("Webcam bağlantısı bekleniyor...")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(480, 360)
        self.webcam_label.setStyleSheet("border: 2px solid #888888; background-color: #222222; color: white;")
        
        # Yaşlandırılmış görüntü (şimdilik boş)
        self.aged_label = QLabel("Yaşlandırılmış görüntü")
        self.aged_label.setAlignment(Qt.AlignCenter)
        self.aged_label.setMinimumSize(480, 360)
        self.aged_label.setStyleSheet("border: 2px solid #888888; background-color: #222222; color: white;")
        
        # Görüntüleri layout'a ekle
        self.video_layout.addWidget(self.webcam_label)
        self.video_layout.addWidget(self.aged_label)
        
        # Kontrol paneli
        self.controls_layout = QHBoxLayout()
        
        # Yaşlandırma seçenekleri grubu
        self.aging_group = QGroupBox("Yaşlandırma Seçenekleri")
        self.aging_layout = QVBoxLayout(self.aging_group)
        
        # Yaşlandırma seçenekleri açılır menüsü
        self.aging_combo = QComboBox()
        self.aging_combo.addItems(["10 Yıl Yaşlandır", "20 Yıl Yaşlandır", "40 Yıl Yaşlandır"])
        self.aging_layout.addWidget(self.aging_combo)
        
        # Kamera kontrolleri grubu
        self.camera_group = QGroupBox("Kamera Kontrolleri")
        self.camera_layout = QVBoxLayout(self.camera_group)
        
        # Kamera başlat/durdur butonu
        self.camera_button = QPushButton("Kamerayı Başlat")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.camera_layout.addWidget(self.camera_button)
        
        # Ekran görüntüsü butonu
        self.screenshot_button = QPushButton("Ekran Görüntüsü Al")
        self.screenshot_button.clicked.connect(self.take_screenshot)
        self.camera_layout.addWidget(self.screenshot_button)
        
        # Kontrol gruplarını ana layout'a ekle
        self.controls_layout.addWidget(self.aging_group)
        self.controls_layout.addWidget(self.camera_group)
        
        # Ana layout'a alt layout'ları ekle
        self.main_layout.addLayout(self.video_layout, 3)  # 3 birim ağırlık
        self.main_layout.addLayout(self.controls_layout, 1)  # 1 birim ağırlık
        
        # Webcam değişkenleri
        self.webcam = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.camera_running = False
        
    def toggle_camera(self):
        """Kamerayı başlatır veya durdurur."""
        if not self.camera_running:
            # Kamerayı başlat
            self.webcam = cv2.VideoCapture(0)  # 0 = varsayılan kamera
            
            if not self.webcam.isOpened():
                QMessageBox.critical(self, "Kamera Hatası", 
                                     "Webcam açılamadı! Kamera bağlantınızı kontrol edin.")
                return
            
            # Kamera açıldıysa
            self.timer.start(30)  # 30ms aralıklarla (yaklaşık 33fps)
            self.camera_running = True
            self.camera_button.setText("Kamerayı Durdur")
            self.webcam_label.setText("Görüntü yükleniyor...")
        else:
            # Kamerayı durdur
            self.timer.stop()
            if self.webcam and self.webcam.isOpened():
                self.webcam.release()
                
            self.camera_running = False
            self.camera_button.setText("Kamerayı Başlat")
            self.webcam_label.setText("Webcam bağlantısı bekleniyor...")
            self.aged_label.setText("Yaşlandırılmış görüntü")
            
    def update_frame(self):
        """Webcam görüntüsünü günceller."""
        ret, frame = self.webcam.read()
        
        if not ret:
            self.timer.stop()
            self.camera_running = False
            self.camera_button.setText("Kamerayı Başlat")
            self.webcam_label.setText("Kamera bağlantısı kesildi!")
            return
        
        # OpenCV BGR -> RGB dönüşümü
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ayna efekti (selfie görünümü)
        frame_rgb = cv2.flip(frame_rgb, 1)
        
        # Görüntüyü QImage'e dönüştür
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        qt_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # QLabel'a görüntüyü göster
        pixmap = QPixmap.fromImage(qt_image)
        self.webcam_label.setPixmap(pixmap.scaled(
            self.webcam_label.width(), 
            self.webcam_label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
        # NOT: Gerçek uygulamada burada yüz tespiti ve yaşlandırma işlemleri yapılacak
        # Şimdilik aynı görüntüyü yaşlandırılmış tarafta da gösterelim
        self.aged_label.setPixmap(pixmap.scaled(
            self.aged_label.width(), 
            self.aged_label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def take_screenshot(self):
        """Ekran görüntüsü alır ve kaydeder."""
        if not self.camera_running:
            QMessageBox.warning(self, "Uyarı", "Ekran görüntüsü almak için önce kamerayı başlatın!")
            return
            
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webcam_screenshot_{timestamp}.png"
        
        # Şu anki pixmap'i kaydet
        if self.webcam_label.pixmap():
            self.webcam_label.pixmap().save(filename, "PNG")
            QMessageBox.information(self, "Bilgi", f"Ekran görüntüsü kaydedildi: {filename}")
        else:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek görüntü bulunamadı!")
    
    def closeEvent(self, event):
        """Uygulama kapatıldığında kaynakları serbest bırakır."""
        if self.webcam and self.webcam.isOpened():
            self.webcam.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern görünüm
    window = WebcamApp()
    window.show()
    sys.exit(app.exec_())