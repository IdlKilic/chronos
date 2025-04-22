import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
                             QGroupBox, QMessageBox, QFileDialog, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Import GAN model
from gan_model import IPCGAN, FaceDetector

class WebcamApp(QMainWindow):
    def __init__(self):
        super().__init__() 
        
        # Ana pencere ayarları
        self.setWindowTitle("Chronos - Yüz Yaşlandırma")
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
        
        # Yaşlandırılmış görüntü
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
        self.aging_combo.currentIndexChanged.connect(self.on_aging_changed)
        self.aging_layout.addWidget(self.aging_combo)
        
        # Model yükleme durumu
        self.model_status = QLabel("Model Durumu: Yüklenmedi")
        self.aging_layout.addWidget(self.model_status)
        
        # Model yükleme butonu
        self.load_model_button = QPushButton("Model Yükle")
        self.load_model_button.clicked.connect(self.load_model)
        self.aging_layout.addWidget(self.load_model_button)
        
        # Yükleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.aging_layout.addWidget(self.progress_bar)
        
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
        
        # Resim yükle butonu
        self.load_image_button = QPushButton("Resim Yükle")
        self.load_image_button.clicked.connect(self.load_image)
        self.camera_layout.addWidget(self.load_image_button)
        
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
        
        # Image processing
        self.current_frame = None
        self.detected_faces = []
        self.current_face = None
        self.current_face_rect = None
        self.aged_face = None
        
        # GAN model
        self.gan_model = None
        self.face_detector = FaceDetector()
        self.current_age = 0  # 10 yıl yaşlandır (default)
        
        # Results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_model(self):
        """GAN modelini yükler."""
        try:
            # Model dosyasını seç
            model_path, _ = QFileDialog.getOpenFileName(
                self, "IPCGAN Model Dosyasını Seç", "", "Model Files (*.pth *.pt);;All Files (*)"
            )
            
            if not model_path:
                return
            
            # İlerleme çubuğunu güncelle
            self.progress_bar.setValue(10)
            
            # CUDA kullanılabilir mi kontrol et
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Model yükleniyor mesajı
            self.model_status.setText(f"Model Durumu: Yükleniyor... ({device} üzerinde)")
            QApplication.processEvents()  # UI'yı güncelle
            
            # Modeli yükle
            self.gan_model = IPCGAN(model_path=model_path, device=device)
            
            # İlerleme çubuğunu güncelle
            self.progress_bar.setValue(100)
            
            # Model durumunu güncelle
            self.model_status.setText(f"Model Durumu: Yüklendi ({device})")
            self.load_model_button.setEnabled(False)
            
        except Exception as e:
            # Hata mesajı
            self.model_status.setText(f"Model Durumu: Hata - {str(e)}")
            QMessageBox.critical(self, "Model Yükleme Hatası", f"Model yüklenirken hata oluştu: {str(e)}")
            self.progress_bar.setValue(0)
    
    def on_aging_changed(self, index):
        """Yaşlandırma seçeneği değiştiğinde."""
        # Yaşlandırma seçeneklerini güncelle
        self.current_age = index
        
        # Eğer yüz tespiti yapılmışsa, yaşlandırma işlemini tekrar yap
        if self.current_face is not None and self.gan_model is not None:
            self.age_current_face()
    
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
        
        # Görüntüyü kaydet
        self.current_frame = frame.copy()
        
        # Ayna efekti (selfie görünümü)
        frame = cv2.flip(frame, 1)
        
        # Yüz tespiti yap
        self.detect_faces(frame)
        
        # Yüz tespit edildiyse ve model yüklüyse yaşlandırma işlemini yap
        if len(self.detected_faces) > 0 and self.gan_model is not None:
            # İlk yüzü al
            face_rect = self.detected_faces[0]
            
            # Yüzü çıkar
            face_img, face_rect = self.face_detector.extract_face(frame, face_rect)
            
            # Yüzü kaydet
            self.current_face = face_img
            self.current_face_rect = face_rect
            
            # Yaşlandırma işlemi
            self.age_current_face()
            
            # Yaşlandırılmış yüzü orijinal çerçeveye yerleştir
            if self.aged_face is not None:
                # Yaşlandırılmış yüzü orijinal boyutuna yeniden boyutlandır
                aged_resized = cv2.resize(self.aged_face, (face_rect[2], face_rect[3]))
                
                # Yaşlandırılmış yüzü göster
                self.display_image(aged_resized, self.aged_label)
        
        # Algılanan yüzleri çerçeve içine al
        for (x, y, w, h) in self.detected_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # OpenCV BGR -> RGB dönüşümü (QImage için)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Görüntüyü göster
        self.display_image(frame_rgb, self.webcam_label)
    
    def detect_faces(self, frame):
        """Görüntüde yüz tespiti yapar."""
        # Gri tonlamaya dönüştür
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüz tespiti
        faces = self.face_detector.detect_faces(frame)
        
        # Tespit edilen yüzleri kaydet
        self.detected_faces = faces
    
    def age_current_face(self):
        """Mevcut yüzü yaşlandırır."""
        if self.current_face is None or self.gan_model is None:
            return
        
        try:
            # Yaşlandırma işlemi
            self.aged_face = self.gan_model.age_face(self.current_face, self.current_age)
        except Exception as e:
            print(f"Yaşlandırma hatası: {str(e)}")
    
    def display_image(self, img, label):
        """Görüntüyü QLabel'a gösterir."""
        # Görüntü RGB formatında değilse dönüştür
        if len(img.shape) == 3 and img.shape[2] == 3:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            # Eğer BGR ise RGB'ye çevir (QImage için)
            if isinstance(img, np.ndarray) and img.shape[2] == 3:
                if not np.array_equal(img[0, 0, :], img[0, 0, ::-1]):  # BGR kontrolü
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Görüntüyü QImage'e dönüştür
        height, width = img.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # QPixmap oluştur ve göster
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(
            label.width(), 
            label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def load_image(self):
        """Disk'ten resim yükler."""
        # Dosya seçme diyaloğu
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Resim Dosyasını Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Resmi yükle
            img = cv2.imread(file_path)
            
            if img is None:
                QMessageBox.warning(self, "Hata", "Resim yüklenemedi!")
                return
            
            # Resmi kaydet
            self.current_frame = img.copy()
            
            # Yüz tespiti
            self.detect_faces(img)
            
            # Yüz tespit edildiyse
            if len(self.detected_faces) > 0:
                # İlk yüzü al
                face_rect = self.detected_faces[0]
                
                # Yüzü çıkar
                face_img, face_rect = self.face_detector.extract_face(img, face_rect)
                
                # Yüzü kaydet
                self.current_face = face_img
                self.current_face_rect = face_rect
                
                # Tespit edilen yüzü çerçeve içine al
                img_with_face = img.copy()
                for (x, y, w, h) in self.detected_faces:
                    cv2.rectangle(img_with_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Görüntüyü göster
                self.display_image(cv2.cvtColor(img_with_face, cv2.COLOR_BGR2RGB), self.webcam_label)
                
                # Model yüklüyse yaşlandırma işlemini yap
                if self.gan_model is not None:
                    self.age_current_face()
                    
                    # Yaşlandırılmış yüzü göster
                    if self.aged_face is not None:
                        self.display_image(cv2.cvtColor(self.aged_face, cv2.COLOR_BGR2RGB), self.aged_label)
            else:
                # Yüz tespit edilmediyse uyarı ver
                self.webcam_label.setText("Yüz tespit edilemedi!")
                self.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.webcam_label)
        
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Resim işlenirken hata oluştu: {str(e)}")
    
    def take_screenshot(self):
        """Ekran görüntüsü alır ve kaydeder."""
        if self.current_frame is None and not self.camera_running:
            QMessageBox.warning(self, "Uyarı", "Ekran görüntüsü almak için önce kamerayı başlatın veya resim yükleyin!")
            return
            
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sonuçlar klasörünü oluştur
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Dosya adları
        original_filename = os.path.join(self.results_dir, f"original_{timestamp}.png")
        aged_filename = os.path.join(self.results_dir, f"aged_{timestamp}.png")
        
        try:
            # Orijinal görüntüyü kaydet
            if self.webcam_label.pixmap():
                self.webcam_label.pixmap().save(original_filename, "PNG")
            
            # Yaşlandırılmış görüntüyü kaydet
            if self.aged_label.pixmap():
                self.aged_label.pixmap().save(aged_filename, "PNG")
                QMessageBox.information(self, "Bilgi", f"Ekran görüntüleri kaydedildi:\n"
                                                     f"Orijinal: {original_filename}\n"
                                                     f"Yaşlandırılmış: {aged_filename}")
            else:
                QMessageBox.information(self, "Bilgi", f"Orijinal görüntü kaydedildi: {original_filename}")
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Ekran görüntüsü kaydedilirken hata oluştu: {str(e)}")
    
    def closeEvent(self, event):
        """Uygulama kapatıldığında kaynakları serbest bırakır."""
        if self.webcam and self.webcam.isOpened():
            self.webcam.release()
        event.accept()