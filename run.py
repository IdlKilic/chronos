import sys
import os
import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from PyQt5.QtWidgets import QApplication, QSplashScreen, QProgressBar, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

from simple_webcam_gui import WebcamApp, AppSingleton


class LoggerSingleton:
    _instance = None
    
    @staticmethod
    def get_instance():
        if LoggerSingleton._instance is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler("application.log"),
                    logging.StreamHandler()
                ]
            )
            LoggerSingleton._instance = logging.getLogger(__name__)
        return LoggerSingleton._instance


# Strategy Pattern: Farklı sistem kontrol stratejileri
class SystemCheckStrategy(ABC):
    @abstractmethod
    def check(self) -> Dict[str, Any]:
        pass


# CUDA kullanılabilirliğini kontrol eden strateji
class CudaCheckStrategy(SystemCheckStrategy):
    def check(self) -> Dict[str, Any]:
        logger = LoggerSingleton.get_instance()
        result = {"name": "CUDA Kontrolü", "status": "Başarısız", "details": [], "passed": False}
        
        try:
            cuda_available = torch.cuda.is_available()
            result["passed"] = cuda_available
            result["status"] = "Başarılı" if cuda_available else "Uyarı"
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
                
                result["details"].append(f"CUDA kullanılabilir: Evet")
                result["details"].append(f"Cihaz sayısı: {device_count}")
                for i, name in enumerate(device_names):
                    result["details"].append(f"Cihaz {i}: {name}")
            else:
                result["details"].append("CUDA kullanılamıyor. CPU üzerinde çalışılacak.")
                
        except Exception as e:
            result["status"] = "Hata"
            result["details"].append(f"CUDA kontrolü sırasında hata: {str(e)}")
            logger.error(f"CUDA kontrolü sırasında hata: {str(e)}")
            
        return result


# Veri klasörünü kontrol eden strateji
class DatasetCheckStrategy(SystemCheckStrategy):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
    
    def check(self) -> Dict[str, Any]:
        logger = LoggerSingleton.get_instance()
        result = {"name": "Veri Seti Kontrolü", "status": "Başarısız", "details": [], "passed": False}
        
        try:
            if not os.path.exists(self.dataset_dir):
                result["details"].append(f"Uyarı: '{self.dataset_dir}' klasörü bulunamadı.")
                return result
            
            # Alt klasörleri kontrol et (001, 002, ...)
            subfolders = [f for f in os.listdir(self.dataset_dir) 
                          if os.path.isdir(os.path.join(self.dataset_dir, f))]
            
            if not subfolders:
                result["details"].append(f"Uyarı: '{self.dataset_dir}' klasöründe alt klasör bulunamadı.")
                return result
            
            # Numaralandırılmış klasörleri kontrol et
            numeric_folders = [f for f in subfolders if f.isdigit()]
            
            result["passed"] = bool(numeric_folders)
            result["status"] = "Başarılı" if result["passed"] else "Uyarı"
            
            result["details"].append(f"Toplam {len(subfolders)} klasör bulundu.")
            result["details"].append(f"Yaş klasörü sayısı: {len(numeric_folders)}")
            
            # Örnek yaş aralıklarını göster
            if numeric_folders:
                numeric_folders.sort()
                min_age = numeric_folders[0]
                max_age = numeric_folders[-1]
                result["details"].append(f"Yaş aralığı: {min_age} - {max_age}")
                
        except Exception as e:
            result["status"] = "Hata"
            result["details"].append(f"Veri seti kontrolü sırasında hata: {str(e)}")
            logger.error(f"Veri seti kontrolü sırasında hata: {str(e)}")
            
        return result


# Model dosyalarını kontrol eden strateji
class ModelCheckStrategy(SystemCheckStrategy):
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
    
    def check(self) -> Dict[str, Any]:
        logger = LoggerSingleton.get_instance()
        result = {"name": "Model Kontrolü", "status": "Başarısız", "details": [], "passed": False}
        
        try:
            # Model klasörünü oluştur (yoksa)
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Model dosyalarını kontrol et
            model_files = [f for f in os.listdir(self.model_dir) 
                          if f.endswith(('.pth', '.pt')) and 
                          os.path.isfile(os.path.join(self.model_dir, f))]
            
            result["passed"] = bool(model_files)
            result["status"] = "Başarılı" if result["passed"] else "Uyarı"
            
            if model_files:
                result["details"].append(f"'{self.model_dir}' klasöründe {len(model_files)} model dosyası bulundu:")
                for model_file in model_files:
                    size_mb = os.path.getsize(os.path.join(self.model_dir, model_file)) / (1024 * 1024)
                    result["details"].append(f"  - {model_file} ({size_mb:.2f} MB)")
            else:
                result["details"].append(f"Uyarı: '{self.model_dir}' klasöründe model dosyası bulunamadı.")
                
        except Exception as e:
            result["status"] = "Hata"
            result["details"].append(f"Model kontrolü sırasında hata: {str(e)}")
            logger.error(f"Model kontrolü sırasında hata: {str(e)}")
            
        return result


# Çıktı klasörünü kontrol eden strateji
class OutputDirCheckStrategy(SystemCheckStrategy):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def check(self) -> Dict[str, Any]:
        logger = LoggerSingleton.get_instance()
        result = {"name": "Çıktı Klasörü Kontrolü", "status": "Başarısız", "details": [], "passed": False}
        
        try:
            # Çıktı klasörünü oluştur (yoksa)
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Klasör yazılabilir mi kontrol et
            is_writable = os.access(self.output_dir, os.W_OK)
            
            result["passed"] = is_writable
            result["status"] = "Başarılı" if is_writable else "Hata"
            
            if is_writable:
                result["details"].append(f"'{self.output_dir}' klasörü oluşturuldu ve yazılabilir.")
            else:
                result["details"].append(f"Hata: '{self.output_dir}' klasörü yazılabilir değil.")
                
        except Exception as e:
            result["status"] = "Hata"
            result["details"].append(f"Çıktı klasörü kontrolü sırasında hata: {str(e)}")
            logger.error(f"Çıktı klasörü kontrolü sırasında hata: {str(e)}")
            
        return result


# Factory Pattern: Sistem kontrol stratejisi oluşturucu
class SystemCheckFactory:
    @staticmethod
    def create_check_strategies(config: Dict[str, Any]) -> List[SystemCheckStrategy]:
        strategies = [
            CudaCheckStrategy(),
            DatasetCheckStrategy(config.get("dataset_dir", "cleandataset")),
            ModelCheckStrategy(config.get("model_dir", "models")),
            OutputDirCheckStrategy(config.get("output_dir", "results"))
        ]
        return strategies


# Chain of Responsibility Pattern: Sistem kontrolü zinciri
class SystemCheckHandler(ABC):
    def __init__(self):
        self._next_handler = None
    
    def set_next(self, handler):
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass


# Temel sistem kontrolü için handler
class BasicSystemCheckHandler(SystemCheckHandler):
    def __init__(self, strategies: List[SystemCheckStrategy]):
        super().__init__()
        self.strategies = strategies
    
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        logger = LoggerSingleton.get_instance()
        logger.info("Temel sistem kontrolü başlatılıyor...")
        
        request["check_results"] = []
        
        for strategy in self.strategies:
            result = strategy.check()
            request["check_results"].append(result)
            logger.info(f"{result['name']}: {result['status']}")
            
            for detail in result["details"]:
                logger.info(f"  - {detail}")
        
        if self._next_handler:
            return self._next_handler.handle(request)
        
        return request


# Kritik sistem kontrolü için handler
class CriticalCheckHandler(SystemCheckHandler):
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        logger = LoggerSingleton.get_instance()
        logger.info("Kritik sistem kontrolü başlatılıyor...")
        
        # Kritik hataları kontrol et
        critical_failures = []
        
        for result in request.get("check_results", []):
            if result["status"] == "Hata" and not result["passed"]:
                critical_failures.append(result["name"])
        
        request["critical_failures"] = critical_failures
        
        if critical_failures:
            logger.error(f"Kritik hatalar tespit edildi: {', '.join(critical_failures)}")
        else:
            logger.info("Kritik hata tespit edilmedi.")
        
        if self._next_handler:
            return self._next_handler.handle(request)
        
        return request


# Observer Pattern: Sistem kontrolü gözlemcisi
class SystemCheckObserver(ABC):
    @abstractmethod
    def update(self, check_results: List[Dict[str, Any]], critical_failures: List[str]):
        pass


# Komut satırına bilgi yazan gözlemci
class ConsoleObserver(SystemCheckObserver):
    def update(self, check_results: List[Dict[str, Any]], critical_failures: List[str]):
        print("\n===== SİSTEM KONTROL SONUÇLARI =====")
        
        for result in check_results:
            status_symbol = "✓" if result["passed"] else "✗"
            print(f"{status_symbol} {result['name']}: {result['status']}")
            
            for detail in result["details"]:
                print(f"  - {detail}")
        
        if critical_failures:
            print("\n⚠️ KRİTİK HATALAR:")
            for failure in critical_failures:
                print(f"  - {failure}")
        
        print("\n" + "=" * 36 + "\n")


# Splash ekranına bilgi yazan gözlemci
class SplashScreenObserver(SystemCheckObserver):
    def __init__(self, splash_screen, progress_bar, status_label):
        self.splash_screen = splash_screen
        self.progress_bar = progress_bar
        self.status_label = status_label
        self.total_checks = 0
        self.current_check = 0
    
    def set_total_checks(self, total: int):
        self.total_checks = total
        self.progress_bar.setMaximum(total)
    
    def update(self, check_results: List[Dict[str, Any]], critical_failures: List[str]):
        # İlerleme çubuğunu güncelle
        self.current_check = len(check_results)
        self.progress_bar.setValue(self.current_check)
        
        if check_results:
            latest_result = check_results[-1]
            self.status_label.setText(f"{latest_result['name']} kontrol ediliyor...")
            
            # Çok uzun zaman alırsa, UI'ı güncelle
            QApplication.processEvents()
        
        # Kritik hatalar varsa bildirimi daha uzun süre göster
        if self.current_check == self.total_checks:
            if critical_failures:
                self.status_label.setText("Kritik hatalar tespit edildi!")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.status_label.setText("Sistem kontrolü tamamlandı!")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
            
            QApplication.processEvents()


# Subject sınıfı - Observer pattern için
class SystemCheckSubject:
    def __init__(self):
        self._observers = []
    
    def register_observer(self, observer: SystemCheckObserver):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer: SystemCheckObserver):
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify_observers(self, check_results: List[Dict[str, Any]], critical_failures: List[str]):
        for observer in self._observers:
            observer.update(check_results, critical_failures)


# Facade Pattern: Tüm sistem kontrolü işlemlerini kapsüller
class SystemCheckFacade:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.subject = SystemCheckSubject()
        self.logger = LoggerSingleton.get_instance()
    
    def register_observer(self, observer: SystemCheckObserver):
        self.subject.register_observer(observer)
        
        # Eğer observer SplashScreenObserver ise toplam kontrol sayısını ayarla
        if isinstance(observer, SplashScreenObserver):
            observer.set_total_checks(len(SystemCheckFactory.create_check_strategies(self.config)))
    
    def perform_checks(self) -> Dict[str, Any]:
        # Kontrol stratejilerini oluştur
        strategies = SystemCheckFactory.create_check_strategies(self.config)
        
        # Kontrol zincirini oluştur
        basic_handler = BasicSystemCheckHandler(strategies)
        critical_handler = CriticalCheckHandler()
        
        basic_handler.set_next(critical_handler)
        
        # Kontrolleri gerçekleştir
        self.logger.info("Sistem kontrolü başlatılıyor...")
        result = basic_handler.handle({})
        
        # Gözlemcileri bilgilendir
        self.subject.notify_observers(
            result.get("check_results", []),
            result.get("critical_failures", [])
        )
        
        return result


# Splash screen sınıfı
class ChronosSplashScreen(QSplashScreen):
    def __init__(self):
        # Splash ekranı için bir pixmap oluştur
        pixmap = QPixmap(400, 300)
        pixmap.fill(Qt.white)
        super().__init__(pixmap)
        
        # Splash ekranı içeriğini oluştur
        layout_widget = QWidget(self)
        layout = QVBoxLayout(layout_widget)
        
        title_label = QLabel("Chronos - Yüz Yaşlandırma")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        
        self.status_label = QLabel("Sistem kontrol ediliyor...")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        layout.addWidget(title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        
        # Widget'ı ortala
        layout_widget.setGeometry((pixmap.width() - 350) // 2, 
                                 (pixmap.height() - 100) // 2, 
                                 350, 100)
    
    def get_progress_bar(self) -> QProgressBar:
        return self.progress_bar
    
    def get_status_label(self) -> QLabel:
        return self.status_label


# Command Pattern: Uygulama başlatma komutu
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


class StartApplicationCommand(Command):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def execute(self):
        logger = LoggerSingleton.get_instance()
        logger.info("Uygulama başlatılıyor...")
        
        # PyQt uygulamasını başlat
        app = QApplication(sys.argv)
        app.setStyle(self.config.get("style", "Fusion"))
        
        # Splash ekranını göster
        splash = ChronosSplashScreen()
        splash.show()
        
        # Sistem kontrolünü gerçekleştir
        system_check = SystemCheckFacade(self.config)
        
        # Gözlemcileri ekle
        console_observer = ConsoleObserver()
        splash_observer = SplashScreenObserver(
            splash, splash.get_progress_bar(), splash.get_status_label())
        
        system_check.register_observer(console_observer)
        system_check.register_observer(splash_observer)
        
        # Sistem kontrolünü başlat
        result = system_check.perform_checks()
        
        # Kritik hatalar varsa uyarı ver ve çık
        if result.get("critical_failures", []):
            logger.error("Kritik hatalar nedeniyle uygulama başlatılamıyor!")
            
            # Kullanıcının splash ekranındaki mesajı görmesi için bekle
            QTimer.singleShot(3000, lambda: sys.exit(1))
            return app.exec_()
        
        # Ana pencereyi göster
        QTimer.singleShot(1500, lambda: self._show_main_window(splash))
        
        # Uygulamayı çalıştır
        return app.exec_()
    
    def _show_main_window(self, splash):
        # AppSingleton'dan WebcamApp örneğini al
        _, window = AppSingleton.get_instance()
        window.show()
        splash.finish(window)


# Configuration Manager - Singleton Pattern
class ConfigManager:
    _instance = None
    
    @staticmethod
    def get_instance():
        if ConfigManager._instance is None:
            ConfigManager._instance = ConfigManager()
        return ConfigManager._instance
    
    def __init__(self):
        if ConfigManager._instance is not None:
            raise Exception("Bu bir Singleton sınıfıdır. get_instance() metodunu kullanın.")
        
        self.config = {
            "dataset_dir": "cleandataset",
            "model_dir": "models",
            "output_dir": "results",
            "style": "Fusion",
            "debug": False
        }
    
    def get_config(self) -> Dict[str, Any]:
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        self.config.update(updates)


# Ana fonksiyon
def main():
<<<<<<< HEAD
    # Konfigürasyon Yöneticisini başlat
    config_manager = ConfigManager.get_instance()
=======
    # PyQt uygulamasını başlatma fonksiyonu
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  
    check_requirements()
    window = WebcamApp()
    window.show()
>>>>>>> 757cc57c4f5abb757f0bc34e9d78f431d3c80716
    
    # Komut Satırı Argümanlarını İşle
    import argparse
    parser = argparse.ArgumentParser(description='Chronos - Yüz Yaşlandırma Uygulaması')
    
    parser.add_argument('--dataset', type=str, help='Veri seti klasörü yolu')
    parser.add_argument('--models', type=str, help='Model klasörü yolu')
    parser.add_argument('--output', type=str, help='Çıktı klasörü yolu')
    parser.add_argument('--style', type=str, help='UI stili (Fusion, Windows, vb.)')
    parser.add_argument('--debug', action='store_true', help='Hata ayıklama modunu etkinleştir')
    
    args = parser.parse_args()
    
    # Argümanları konfigürasyona uygula
    updates = {}
    if args.dataset:
        updates["dataset_dir"] = args.dataset
    if args.models:
        updates["model_dir"] = args.models
    if args.output:
        updates["output_dir"] = args.output
    if args.style:
        updates["style"] = args.style
    if args.debug:
        updates["debug"] = True
    
    config_manager.update_config(updates)
    
    # Uygulama Komutunu Oluştur ve Çalıştır
    command = StartApplicationCommand(config_manager.get_config())
    exit_code = command.execute()
    
    # Çıkış kodunu döndür
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
