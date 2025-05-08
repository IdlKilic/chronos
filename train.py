import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
   
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.res_block(x)

class Generator(nn.Module):
    """Yaşlandırma Üretici (Generator) Modeli"""
    def __init__(self, conv_dim=64, c_dim=101, num_residual_blocks=6):
        super(Generator, self).__init__()
        
        # Encoder katmanları
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, conv_dim, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(True)
        )
        
        # Downsampling katmanları
        self.downsampling = nn.ModuleList()
        curr_dim = conv_dim
        for i in range(2):
            self.downsampling.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim*2),
                nn.ReLU(True)
            ))
            curr_dim = curr_dim * 2
        
        # Residual bloklar
        self.residual_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(curr_dim))
        
        # Upsampling katmanları
        self.upsampling = nn.ModuleList()
        for i in range(2):
            self.upsampling.append(nn.Sequential(
                nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim//2),
                nn.ReLU(True)
            ))
            curr_dim = curr_dim // 2
        
        # Çıkış katmanı
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, 3, kernel_size=7, padding=0, bias=False),
            nn.Tanh()
        )
        
        # Yaş katmanı
        self.age_mlp = nn.Sequential(
            nn.Linear(c_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True)
        )

    def forward(self, x, c):
       
        c_embed = self.age_mlp(c)
        c_embed = c_embed.view(c_embed.size(0), c_embed.size(1), 1, 1)
        c_embed = c_embed.repeat(1, 1, x.size(2), x.size(3))
        
        # Encoder
        x = self.encoder(x)
        
        # Downsampling
        for down_layer in self.downsampling:
            x = down_layer(x)
        
        # Residual bloklar
        for res_block in self.residual_blocks:
            x = res_block(x)
        
        # Upsampling
        for up_layer in self.upsampling:
            x = up_layer(x)
        
        # Çıkış katmanı
        x = self.output_layer(x)
        
        # Yaş bilgisini ekle
        x = x + 0.1 * c_embed
        
        return x

class FaceDetector:
    """Yüz Algılama Sınıfı"""
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, img, min_neighbors=5):
        """Görüntüdeki yüzleri algılar"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        return faces
    
    def extract_face(self, img, face_rect, padding=0.2):
        """Algılanan yüzü keser ve çerçeve bilgilerini döndürür"""
        x, y, w, h = face_rect
        
        # Yüz etrafına padding ekle
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img.shape[1], x + w + pad_w)
        y2 = min(img.shape[0], y + h + pad_h)
        
        face_img = img[y1:y2, x1:x2]
        padded_rect = (x1, y1, x2-x1, y2-y1)
        
        return face_img, padded_rect

class IPCGAN:
    """Yaşlandırma Modeli Sınıfı"""
    def __init__(self, model_path=None, device='cpu', image_size=128):
        self.device = torch.device(device)
        self.image_size = image_size
        
        # Yaş gruplarını tanımla
        self.age_groups = {
            0: 10,  # 10 yıl yaşlandır
            1: 20,  # 20 yıl yaşlandır
            2: 40   # 40 yıl yaşlandır
        }
        
        # Generator modelini yükle
        self.G = Generator(c_dim=5).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.G.eval()
        
        self.mean = torch.FloatTensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.FloatTensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
    
    def load_model(self, path):
        """Model ağırlıklarını yükler"""
        print(f"Model yükleniyor: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'generator' in checkpoint:
            self.G.load_state_dict(checkpoint['generator'])
        else:
            self.G.load_state_dict(checkpoint)
        
        print("Model başarıyla yüklendi")
    
    def estimate_age(self, face_img):
        """Yüzde görünen yaşı basit bir şekilde tahmin eder"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
   
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        est_age = int(20 + brightness / 10 + contrast / 5)
        est_age = max(1, min(est_age, 100))
        
        return est_age
    
    def preprocess_image(self, img):
        """Görüntüyü model için hazırlar"""
        # Boyutlandırma
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # BGR -> RGB dönüşümü
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalleştirme
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # NumPy -> Torch dönüşümü
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img.unsqueeze(0)
        
        return img.to(self.device)
    
    def postprocess_image(self, tensor):
        """Model çıktısını görüntüye dönüştürür"""
        tensor = tensor.detach().cpu()
        
        # Normalleştirmeyi geri al
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        

        img = tensor.numpy()[0].transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        
        # RGB -> BGR dönüşümü
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def create_age_vector(self, age):
        """Yaş bilgisini model için vektör haline getirir"""
        age_vector = torch.zeros(5)
        
        if age < 20:
            age_vector[0] = 1.0
        elif age < 40:
            age_vector[1] = 1.0
        elif age < 60:
            age_vector[2] = 1.0
        elif age < 80:
            age_vector[3] = 1.0
        else:
            age_vector[4] = 1.0
            
        return age_vector
    
    def age_face(self, img, target_age_category):
        """Yüzü yaşlandırır"""
        self.G.eval()
        
        with torch.no_grad():
            # Görüntüyü hazırla
            x = self.preprocess_image(img)
            
            # Mevcut yaşı tahmin et
            current_age = self.estimate_age(img)
            print(f"Tahmin edilen mevcut yaş: {current_age}")
            
            # Hedef yaşı hesapla
            years_to_add = self.age_groups[target_age_category]
            target_age = min(100, current_age + years_to_add)
            print(f"Hedef yaş: {target_age} ({years_to_add} yıl ekleniyor)")
            
            # Yaş vektörünü oluştur
            c = self.create_age_vector(target_age)
            c = c.unsqueeze(0).to(self.device)
            
            # Yaşlandırma işlemini gerçekleştir
            aged_face = self.G(x, c)
            
            # Sonucu işle
            aged_face = self.postprocess_image(aged_face)
            
            return aged_face
