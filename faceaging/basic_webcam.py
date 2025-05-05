import sys
import cv2
import numpy as np

from PyQt5.QtCore import Qt , QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QMainWindow, QBoxLayout, \
    QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap, QImage


class WebCamApp(QMainWindow):
    """ QmainWindow dan miras alinarak webcam uygulamasının ana sınıfı oluşturuluyor"""
    def __init__(self):
        """sınıfın başlatıcı fonksiyonu pencere ve bileşenler burada oluşturulacak"""
        #ana sınıf başlatıcısı
        super().__init__()

        self.setWindowTitle("Basit webcam uygulaması")
        self.setGeometry(300, 300, 500, 500)

        self.setup_ui()
        self.setup_variables()

        self.start_webcam()

    def setup_ui(self):
        """user interface hazırlandığı kısım """
        #ana widget oluşturuluyor
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        #layout oluşturuluyor
        self.layout = QVBoxLayout(self.central_widget)

        #webcam görüntüsünü gösterecek label
        self.webcam_label = QLabel("Webcam Bağlantısı Bekleniyor")
        self.webcam_label.setAlignment(Qt.AlignCenter)

        #label ekleniyor
        self.layout.addWidget(self.webcam_label)

    def setup_variables(self):
        """ kamera zamanlayıcı gibi nesnelerin hazırlandığı method"""
        #webcam değişkenini none olarak başlatıyoruz
        self.webcam = None

        #zamanlayıcı oluşturuyoruz
        self.timer = QTimer(self)
        #zamanlayıcı her tetiklendiğinde frame güncellenecek method çağırılacak
        self.timer.timeout.connect(self.update_frame)
        #mevcut kare
        self.current_frame = None

    def start_webcam(self):
        """webcami başlatan method"""
        # 0 parametresi ile varsayılan kamera açılır videocapture ile
        self.webcam = cv2.VideoCapture(0)

        #kamera açıldı mı kontrolü
        if self.webcam.isOpened():
            self.timer.start(30)
        else:
            self.webcam_label.setText("Hata webcam bağlantısı kurulmadı")

    def update_frame(self):
        """kamera görüntüsü inceleyen method, timer ile düzenli aralıkla ile çağırılır"""
        #kameradan bir kare okur
        ret, frame = self.webcam.read()

        if ret:
            #görüntüyü kaydediyoruz
            self.current_frame = frame.copy()
            #ayna görüntüsü oluşturuyoruz, selfie gibi gözüksün diye
            frame = cv2.flip(frame,1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #numpy dizisini QImage'e
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            #QImage i QPixmap e
            pixmap = QPixmap.fromImage(q_img)

            self.webcam_label.setPixmap(pixmap.scaled(
                self.webcam_label.width(),
                self.webcam_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        else:
            self.webcam_label.setText("Hata kamera görüntüsü alınmadı")
            self.timer.stop()

    def closeEvent(self, event):
        """uygulamayı kapatmak için method, kaynaklar serbest bırakılır"""

        self.timer.stop()

        if self.webcam and self.webcam.isOpened():
            self.webcam.release()

        event.accept()

if __name__ == "__main__":
    #QApplication nesnesi oluşturuyoruz
    app = QApplication(sys.argv)
    #fusion teması kullanıyoruz
    app.setStyle("Fusion")
    #uygulama penceresi oluşturuyoruz
    window = WebCamApp()
    window.show()
    sys.exit(app.exec_())






