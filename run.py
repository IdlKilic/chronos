#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
from simple_webcam_gui import WebcamApp
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  

    window = WebcamApp()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
