#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yüz Yaşlandırma Uygulamasını Başlatan Ana Dosya
"""

import sys
from simple_webcam_gui import WebcamApp
from PyQt5.QtWidgets import QApplication

def main():
    """Uygulamayı başlatır"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern bir görünüm için
    
    # Ana uygulama penceresini oluştur
    window = WebcamApp()
    window.show()
    
    # Uygulamayı çalıştır
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
