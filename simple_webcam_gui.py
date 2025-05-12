import sys
import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Callable

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
                             QGroupBox, QMessageBox, QFileDialog, QProgressBar,
                             QFrame, QSplitter)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QImage, QPixmap

# Paste modülünü içe aktar
from paste import IPCGAN, FaceDetector


# Observer Pattern için gerekli arayüz
class Observer(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass


# Subject interface for the Observer pattern
class Subject(ABC):
    @abstractmethod
    def register_observer(self, observer: Observer):
        pass
    
    @abstractmethod
    def remove_observer(self, observer: Observer):
        pass
    
    @abstractmethod
    def notify_observers(self, *args, **kwargs):
        pass


# Strategy Pattern için yaşlandırma stratejisi arabirimi
class AgingStrategy(ABC):
    @abstractmethod
    def age_face(self, face_img, detector, model):
        pass


# Farklı yaşlandırma stratejileri
class AgeBy10Years(AgingStrategy):
    def age_face(self, face_img, detector, model):
        return model.age_face(face_img, 0)  # 0 indeksi +10 yıl


class AgeBy20Years(AgingStrategy):
    def age_face(self, face_img, detector, model):
        return model.age_face(face_img, 1)  # 1 indeksi +20 yıl


class AgeBy30Years(AgingStrategy):
    def age_face(self, face_img, detector, model):
        return model.age_face(face_img, 2)  # 2 indeksi +30 yıl


class AgeBy40Years(AgingStrategy):
    def age_face(self, face_img, detector, model):
        return model.age_face(face_img, 3)  # 3 indeksi +40 yıl


class AgeBy50Years(AgingStrategy):
    def age_face(self, face_img, detector, model):
        return model.age_face(face_img, 4)  # 4 indeksi +50 yıl


# Factory Pattern için strateji fabrikası
class AgingStrategyFactory:
    @staticmethod
    def create_strategy(years: int) -> AgingStrategy:
        strategies = {
            10: AgeBy10Years(),
            20: AgeBy20Years(),
            30: AgeBy30Years(),
            40: AgeBy40Years(),
            50: AgeBy50Years()
        }
        return strategies.get(years, AgeBy10Years())  # Varsayılan olarak 10 yıl


# Facade Pattern için kamera işlemlerini yöneten sınıf
class CameraManager:
    def __init__(self, frame_callback: Callable):
        self.webcam = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.frame_callback = frame_callback
        self.is_running = False
    
    def start(self) -> bool:
        if self.is_running:
            return True
            
        self.webcam = cv2.VideoCapture(0)
        
        if not self.webcam.isOpened():
            return False
        
        self.timer.start(30)  # ~33 FPS
        self.is_running = True
        return True
    
    def stop(self) -> None:
        self.timer.stop()
        if self.webcam and self.webcam.isOpened():
            self.webcam.release()
        self.is_running = False
    
    def is_camera_running(self) -> bool:
        return self.is_running
    
    def _update_frame(self) -> None:
        if not self.webcam:
            return
            
        ret, frame = self.webcam.read()
        
        if not ret:
            self.stop()
            return
        
        # Ayna efekti
        frame = cv2.flip(frame, 1)
        
        # Callback fonksiyonunu çağır
        self.frame_callback(frame)


# Command Pattern için komut arabirimi
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


# Yaşlandırma komutu
class AgeFaceCommand(Command):
    def __init__(self, processor, strategy: AgingStrategy):
        self.processor = processor
        self.strategy = strategy
    
    def execute(self):
        self.processor.set_aging_strategy(self.strategy)
        self.processor.process_current_face()


# Ekran görüntüsü alma komutu
class TakeScreenshotCommand(Command):
    def __init__(self, app):
        self.app = app
    
    def execute(self):
        self.app.save_screenshots()


# Kamera başlatma/durdurma komutu
class ToggleCameraCommand(Command):
    def __init__(self, app):
        self.app = app
    
    def execute(self):
        self.app.toggle_camera()


# Resim yükleme komutu
class LoadImageCommand(Command):
    def __init__(self, app):
        self.app = app
    
    def execute(self):
        self.app.load_image_from_file()


# Model yükleme komutu
class LoadModelCommand(Command):
    def __init__(self, app):
        self.app = app
    
    def execute(self):
        self.app.load_model_from_file()


# Mediator Pattern - UI bileşenleri arasında iletişimi yönetir
class UIMediator:
    def __init__(self):
        self.components = {}
    
    def register_component(self, name: str, component: Any) -> None:
        self.components[name] = component
    
    def notify(self, sender: str, event: str, data: Any = None) -> None:
        # Özel olayları işle
        if event == "model_loaded":
            if "model_status" in self.components:
                self.components["model_status"].setText(f"Model Durumu: {data}")
            if "progress_bar" in self.components:
                self.components["progress_bar"].setValue(100 if data == "Yüklendi" else 0)
            if "age_buttons" in self.components:
                for button in self.components["age_buttons"]:
                    button.setEnabled(data == "Yüklendi")
        
        elif event == "camera_toggled":
            if "camera_button" in self.components:
                self.components["camera_button"].setText(
                    "Kamerayı Durdur" if data else "Kamerayı Başlat")
            if "webcam_label" in self.components:
                if not data:  # Kamera kapalıysa
                    self.components["webcam_label"].setText("Webcam bağlantısı bekleniyor...")
        
        elif event == "face_aged":
            if "aged_label" in self.components and data is not None:
                aged_img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                self._display_image(aged_img, self.components["aged_label"])
        
        elif event == "face_detected":
            if "original_face_label" in self.components and data is not None:
                face_img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                self._display_image(face_img, self.components["original_face_label"])
    
    def _display_image(self, img: np.ndarray, label: QLabel) -> None:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
            
        height, width = img.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(
            label.width(), 
            label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))


# Görüntü işleme sınıfı
class ImageProcessor(Subject):
    def __init__(self, mediator: UIMediator):
        self.face_detector = FaceDetector()
        self.gan_model = None
        self.current_frame = None
        self.current_face = None
        self.current_face_rect = None
        self.aging_strategy = AgeBy10Years()  # Varsayılan strateji
        self.mediator = mediator
        self.observers = []
    
    def register_observer(self, observer: Observer) -> None:
        if observer not in self.observers:
            self.observers.append(observer)
    
    def remove_observer(self, observer: Observer) -> None:
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, *args, **kwargs) -> None:
        for observer in self.observers:
            observer.update(*args, **kwargs)
    
    def set_model(self, model: IPCGAN) -> None:
        self.gan_model = model
        self.notify_observers("model_updated", self.gan_model)
    
    def set_aging_strategy(self, strategy: AgingStrategy) -> None:
        self.aging_strategy = strategy
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        # Görüntüyü kaydet
        self.current_frame = frame.copy()
        
        # Yüz tespiti
        detected_faces = self.face_detector.detect_faces(frame)
        
        # Algılanan yüzleri çerçeve içine al
        for (x, y, w, h) in detected_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Yüz tespit edildiyse ve model yüklüyse yaşlandırma işlemini yap
        if detected_faces and self.gan_model is not None:
            # İlk yüzü al
            face_rect = detected_faces[0]
            
            # Yüzü çıkar
            face_img, face_rect = self.face_detector.extract_face(frame, face_rect)
            
            # Yüzü kaydet
            self.current_face = face_img
            self.current_face_rect = face_rect
            
            # Mediator'a yüz tespit edildiğini bildir
            self.mediator.notify("processor", "face_detected", self.current_face)
            
            # Yaşlandırma işlemini gerçekleştir
            self.process_current_face()
        
        return frame, detected_faces
    
    def process_current_face(self) -> None:
        if self.current_face is None or self.gan_model is None:
            return
        
        try:
            # Strateji desenini kullanarak yaşlandırma işlemi yap
            aged_face = self.aging_strategy.age_face(
                self.current_face, self.face_detector, self.gan_model)
            
            # Mediator'a yaşlandırılmış yüzü bildir
            self.mediator.notify("processor", "face_aged", aged_face)
            
            # Observer'lara bildir
            self.notify_observers("face_aged", aged_face)
            
        except Exception as e:
            print(f"Yaşlandırma hatası: {str(e)}")
    
    def load_and_process_image(self, file_path: str) -> bool:
        try:
            img = cv2.imread(file_path)
            
            if img is None:
                return False
            
            # Resmi işle
            processed_img, detected_faces = self.process_frame(img)
            
            return bool(detected_faces)
            
        except Exception as e:
            print(f"Resim işleme hatası: {str(e)}")
            return False


# UI Bileşenleri
class ImageDisplay(QWidget):
    def __init__(self, title: str, mediator: UIMediator, component_name: str):
        super().__init__()
        self.mediator = mediator
        
        self.layout = QVBoxLayout(self)
        
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold;")
        
        self.image_label = QLabel("Görüntü bekleniyor...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setStyleSheet(
            "border: 2px solid #888888; background-color: #222222; color: white;")
        
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.image_label)
        
        # Mediator'a kaydol
        mediator.register_component(component_name, self.image_label)


class AgingControls(QGroupBox):
    def __init__(self, mediator: UIMediator, commands: Dict[str, Command]):
        super().__init__("Yaşlandırma Seçenekleri")
        self.mediator = mediator
        self.commands = commands
        
        self.layout = QVBoxLayout(self)
        
        # Yaşlandırma butonları
        self.age_buttons_layout = QHBoxLayout()
        self.age_buttons = []
        
        for years in [10, 20, 30, 40, 50]:
            button = QPushButton(f"+{years} Yaş")
            button.setEnabled(False)  # Model yüklenene kadar devre dışı
            button.clicked.connect(lambda checked, y=years: self.on_age_button_clicked(y))
            self.age_buttons.append(button)
            self.age_buttons_layout.addWidget(button)
        
        self.layout.addLayout(self.age_buttons_layout)
        
        # Mediator'a butonları kaydol
        mediator.register_component("age_buttons", self.age_buttons)
        
        # Model durumu
        self.model_status = QLabel("Model Durumu: Yüklenmedi")
        self.layout.addWidget(self.model_status)
        
        # Mediator'a model durumunu kaydol
        mediator.register_component("model_status", self.model_status)
        
        # Model yükleme butonu
        self.load_model_button = QPushButton("Model Yükle")
        self.load_model_button.clicked.connect(self.on_load_model_clicked)
        self.layout.addWidget(self.load_model_button)
        
        # Yükleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)
        
        # Mediator'a progress bar'ı kaydol
        mediator.register_component("progress_bar", self.progress_bar)
    
    def on_age_button_clicked(self, years: int) -> None:
        command_key = f"age_{years}"
        if command_key in self.commands:
            self.commands[command_key].execute()
    
    def on_load_model_clicked(self) -> None:
        if "load_model" in self.commands:
            self.commands["load_model"].execute()


class CameraControls(QGroupBox):
    def __init__(self, mediator: UIMediator, commands: Dict[str, Command]):
        super().__init__("Kamera Kontrolleri")
        self.mediator = mediator
        self.commands = commands
        
        self.layout = QVBoxLayout(self)
        
        # Kamera başlat/durdur butonu
        self.camera_button = QPushButton("Kamerayı Başlat")
        self.camera_button.clicked.connect(self.on_camera_button_clicked)
        self.layout.addWidget(self.camera_button)
        
        # Mediator'a kamera butonunu kaydol
        mediator.register_component("camera_button", self.camera_button)
        
        # Ekran görüntüsü butonu
        self.screenshot_button = QPushButton("Ekran Görüntüsü Al")
        self.screenshot_button.clicked.connect(self.on_screenshot_button_clicked)
        self.layout.addWidget(self.screenshot_button)
        
        # Resim yükle butonu
        self.load_image_button = QPushButton("Resim Yükle")
        self.load_image_button.clicked.connect(self.on_load_image_button_clicked)
        self.layout.addWidget(self.load_image_button)
    
    def on_camera_button_clicked(self) -> None:
        if "toggle_camera" in self.commands:
            self.commands["toggle_camera"].execute()
    
    def on_screenshot_button_clicked(self) -> None:
        if "take_screenshot" in self.commands:
            self.commands["take_screenshot"].execute()
    
    def on_load_image_button_clicked(self) -> None:
        if "load_image" in self.commands:
            self.commands["load_image"].execute()


# Ana uygulama sınıfı
class WebcamApp(QMainWindow, Observer):
    def __init__(self):
        super().__init__()
        
        # Mediator oluştur
        self.mediator = UIMediator()
        
        # Görüntü işlemcisi oluştur
        self.processor = ImageProcessor(self.mediator)
        self.processor.register_observer(self)
        
        # Kamera yöneticisi oluştur
        self.camera_manager = CameraManager(self.on_frame_received)
        
        # Komutlar oluştur
        self.commands = self._create_commands()
        
        # UI bileşenlerini başlat
        self.init_ui()
        
        # Sonuçlar klasörü
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _create_commands(self) -> Dict[str, Command]:
        commands = {
            "toggle_camera": ToggleCameraCommand(self),
            "take_screenshot": TakeScreenshotCommand(self),
            "load_image": LoadImageCommand(self),
            "load_model": LoadModelCommand(self)
        }
        
        # Yaşlandırma komutları
        strategies = {
            10: AgeBy10Years(),
            20: AgeBy20Years(),
            30: AgeBy30Years(),
            40: AgeBy40Years(),
            50: AgeBy50Years()
        }
        
        for years, strategy in strategies.items():
            commands[f"age_{years}"] = AgeFaceCommand(self.processor, strategy)
        
        return commands
    
    def init_ui(self) -> None:
        self.setWindowTitle("Chronos - Yüz Yaşlandırma")
        self.setGeometry(100, 100, 1000, 600)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Görüntü alanı
        self.image_area = QSplitter(Qt.Horizontal)
        
        # Orijinal webcam görüntüsü
        self.webcam_display = ImageDisplay(
            "Orijinal Görüntü", self.mediator, "webcam_label")
        self.image_area.addWidget(self.webcam_display)
        
        # Algılanan yüz görüntüsü
        self.face_display = ImageDisplay(
            "Algılanan Yüz", self.mediator, "original_face_label")
        self.image_area.addWidget(self.face_display)
        
        # Yaşlandırılmış görüntü
        self.aged_display = ImageDisplay(
            "Yaşlandırılmış Görüntü", self.mediator, "aged_label")
        self.image_area.addWidget(self.aged_display)
        
        # Kontrol alanı
        self.controls_layout = QHBoxLayout()
        
        # Yaşlandırma kontrolleri
        self.aging_controls = AgingControls(self.mediator, self.commands)
        self.controls_layout.addWidget(self.aging_controls)
        
        # Kamera kontrolleri
        self.camera_controls = CameraControls(self.mediator, self.commands)
        self.controls_layout.addWidget(self.camera_controls)
        
        # Ana layout'a ekle
        self.main_layout.addWidget(self.image_area, 3)  # 3 birim ağırlık
        self.main_layout.addLayout(self.controls_layout, 1)  # 1 birim ağırlık
    
    def update(self, event_type: str, data: Any) -> None:
        # Observer pattern - processor'dan gelen güncellemeleri işle
        if event_type == "model_updated":
            # Model güncellendiğinde UI'ı güncelle
            model_status = "Yüklendi" if data is not None else "Yüklenmedi"
            self.mediator.notify("app", "model_loaded", model_status)
    
    def on_frame_received(self, frame: np.ndarray) -> None:
        # Kameradan gelen çerçeveyi işle
        processed_frame, _ = self.processor.process_frame(frame)
        
        # İşlenmiş çerçeveyi göster
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        height, width = processed_rgb.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(processed_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Mediator aracılığıyla görüntüyü güncelle
        webcam_label = self.mediator.components.get("webcam_label")
        if webcam_label:
            pixmap = QPixmap.fromImage(q_img)
            webcam_label.setPixmap(pixmap.scaled(
                webcam_label.width(), 
                webcam_label.height(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
    
    def toggle_camera(self) -> None:
        if self.camera_manager.is_camera_running():
            self.camera_manager.stop()
            self.mediator.notify("app", "camera_toggled", False)
        else:
            success = self.camera_manager.start()
            if success:
                self.mediator.notify("app", "camera_toggled", True)
            else:
                QMessageBox.critical(self, "Kamera Hatası", 
                                    "Webcam açılamadı! Kamera bağlantınızı kontrol edin.")
    
    def load_model_from_file(self) -> None:
        try:
            # Model dosyasını seç
            model_path, _ = QFileDialog.getOpenFileName(
                self, "Model Dosyasını Seç", "models", "Model Files (*.pth *.pt);;All Files (*)"
            )
            
            if not model_path:
                return
            
            # Progress bar'ı güncelle
            progress_bar = self.mediator.components.get("progress_bar")
            if progress_bar:
                progress_bar.setValue(10)
            
            # Model durumunu güncelle
            model_status = self.mediator.components.get("model_status")
            if model_status:
                model_status.setText("Model Durumu: Yükleniyor...")
            
            QApplication.processEvents()  # UI'ı güncelle
            
            # CUDA kullanılabilir mi kontrol et
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Modeli yükle
            self.gan_model = IPCGAN(model_path=model_path, device=device)
            
            # İşlemciye modeli bildir
            self.processor.set_model(self.gan_model)
            
            # UI'ı güncelle
            self.mediator.notify("app", "model_loaded", f"Yüklendi ({device})")
            
            # Load model butonunu devre dışı bırak
            self.aging_controls.load_model_button.setEnabled(False)
            
        except Exception as e:
            # Hata mesajı
            self.mediator.notify("app", "model_loaded", f"Hata - {str(e)}")
            QMessageBox.critical(self, "Model Yükleme Hatası", 
                                f"Model yüklenirken hata oluştu: {str(e)}")
            
            progress_bar = self.mediator.components.get("progress_bar")
            if progress_bar:
                progress_bar.setValue(0)
    
    def load_image_from_file(self) -> None:
        # Dosya seçme diyaloğu
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Resim Dosyasını Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Resmi işle
        success = self.processor.load_and_process_image(file_path)
        
        if not success:
            QMessageBox.warning(self, "Uyarı", "Resimde yüz tespit edilemedi!")
    
    def save_screenshots(self) -> None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Dosya adları
        original_filename = os.path.join(self.results_dir, f"original_{timestamp}.png")
        face_filename = os.path.join(self.results_dir, f"face_{timestamp}.png")
        aged_filename = os.path.join(self.results_dir, f"aged_{timestamp}.png")
        
        try:
            # Görüntüleri kaydet
            saved_files = []
            
            webcam_label = self.mediator.components.get("webcam_label")
            if webcam_label and webcam_label.pixmap():
                webcam_label.pixmap().save(original_filename, "PNG")
                saved_files.append(f"Orijinal: {original_filename}")
            
            face_label = self.mediator.components.get("original_face_label")
            if face_label and face_label.pixmap():
                face_label.pixmap().save(face_filename, "PNG")
                saved_files.append(f"Yüz: {face_filename}")
            
            aged_label = self.mediator.components.get("aged_label")
            if aged_label and aged_label.pixmap():
                aged_label.pixmap().save(aged_filename, "PNG")
                saved_files.append(f"Yaşlandırılmış: {aged_filename}")
            
            # Bilgi mesajı
            if saved_files:
                QMessageBox.information(self, "Bilgi", 
                                       f"Ekran görüntüleri kaydedildi:\n" + "\n".join(saved_files))
            else:
                QMessageBox.warning(self, "Uyarı", "Kaydedilecek görüntü bulunamadı!")
                
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", 
                               f"Ekran görüntüsü kaydedilirken hata oluştu: {str(e)}")
    
    def closeEvent(self, event) -> None:
        # Kamera varsa kapat
        self.camera_manager.stop()
        event.accept()


# Singleton pattern ile tek bir uygulama instance'ı oluştur
class AppSingleton:
    _instance = None
    
    @staticmethod
    def get_instance():
        if AppSingleton._instance is None:
            app = QApplication(sys.argv)
            window = WebcamApp()
            AppSingleton._instance = (app, window)
        return AppSingleton._instance


# Ana fonksiyon
def main():
    app, window = AppSingleton.get_instance()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()