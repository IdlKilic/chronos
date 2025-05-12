import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

<<<<<<< HEAD
# Paste modülündeki Generator sınıfını içe aktarıyoruz
from paste import Generator, ResidualBlock
=======
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
>>>>>>> 757cc57c4f5abb757f0bc34e9d78f431d3c80716

# Discriminator sınıfını tanımlıyoruz
class Discriminator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5):
        super(Discriminator, self).__init__()
        
        # Giriş katmanı
        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        
        # Downsampling katmanları
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.InstanceNorm2d(conv_dim*2)
        
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.InstanceNorm2d(conv_dim*4)
        
        self.conv4 = nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.InstanceNorm2d(conv_dim*8)
        
        # Yaş bilgisi işleme katmanı
        self.age_conv = nn.Sequential(
            nn.Linear(c_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 16*16)
        )
<<<<<<< HEAD
=======

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
>>>>>>> 757cc57c4f5abb757f0bc34e9d78f431d3c80716
        
        # Çıkış katmanı
        self.conv5 = nn.Conv2d(conv_dim*8 + 1, 1, kernel_size=4, stride=1, padding=1)
        
        # Aktivasyon fonksiyonu
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x, c):
        h = self.leaky_relu(self.conv1(x))
        
        h = self.conv2(h)
        h = self.conv2_bn(h)
        h = self.leaky_relu(h)
        
        h = self.conv3(h)
        h = self.conv3_bn(h)
        h = self.leaky_relu(h)
        
        h = self.conv4(h)
        h = self.conv4_bn(h)
        h = self.leaky_relu(h)
        
        # Yaş bilgisini işle
        batch_size = x.size(0)
        c_age = self.age_conv(c)
        c_age = c_age.view(batch_size, 1, 16, 16)
        
        # Yaş bilgisini feature map ile birleştir
        h = torch.cat([h, c_age], dim=1)
        
        # Çıkış katmanı
        out = self.conv5(h)
        
        return out

# Yaş sınıflandırıcısı
class AgeClassifier(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5):
        super(AgeClassifier, self).__init__()
        
        # Giriş katmanı
        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        
        # Downsampling katmanları
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.InstanceNorm2d(conv_dim*2)
        
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.InstanceNorm2d(conv_dim*4)
        
        self.conv4 = nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.InstanceNorm2d(conv_dim*8)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Sınıflandırma katmanı
        self.classifier = nn.Linear(conv_dim*8, c_dim)
        
        # Aktivasyon fonksiyonu
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        h = self.leaky_relu(self.conv1(x))
        
        h = self.conv2(h)
        h = self.conv2_bn(h)
        h = self.leaky_relu(h)
        
        h = self.conv3(h)
        h = self.conv3_bn(h)
        h = self.leaky_relu(h)
        
        h = self.conv4(h)
        h = self.conv4_bn(h)
        h = self.leaky_relu(h)
        
        # Global Average Pooling
        h = self.gap(h)
        h = h.view(h.size(0), -1)
        
        # Sınıflandırma
        out = self.classifier(h)
        
        return out

# Veri seti sınıfı
class FaceAgingDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.age_dirs = sorted([d for d in os.listdir(root_dir) 
                                if os.path.isdir(os.path.join(root_dir, d))])
        
        self.image_paths = []
        self.ages = []
        
        for age_dir in self.age_dirs:
            try:
                age = int(age_dir)
                dir_path = os.path.join(root_dir, age_dir)
                
                for img_name in os.listdir(dir_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(dir_path, img_name))
                        self.ages.append(age)
            except ValueError:
                # Klasör adı sayı değilse atla
                continue
        
        # Veri setini bölmek için
        total = len(self.image_paths)
        indices = np.random.permutation(total)
        
        if mode == 'train':
            split_idx = int(total * 0.8)
            self.image_paths = [self.image_paths[i] for i in indices[:split_idx]]
            self.ages = [self.ages[i] for i in indices[:split_idx]]
        elif mode == 'val':
            start_idx = int(total * 0.8)
            end_idx = int(total * 0.9)
            self.image_paths = [self.image_paths[i] for i in indices[start_idx:end_idx]]
            self.ages = [self.ages[i] for i in indices[start_idx:end_idx]]
        else:  # test
            start_idx = int(total * 0.9)
            self.image_paths = [self.image_paths[i] for i in indices[start_idx:]]
            self.ages = [self.ages[i] for i in indices[start_idx:]]
        
        print(f"{mode} veri seti: {len(self.image_paths)} görüntü")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        age = self.ages[idx]
        
        # Görüntüyü yükle
        img = cv2.imread(img_path)
        if img is None:
            # Görüntü yüklenemezse veri setinden başka bir görüntü döndür
            return self.__getitem__((idx + 1) % len(self))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Dönüşümleri uygula
        if self.transform:
            img = self.transform(img)
        
        # Yaş kategorisini belirle (5 kategori: 0-19, 20-39, 40-59, 60-79, 80+)
        age_category = min(age // 20, 4)
        age_vector = torch.zeros(5)
        age_vector[age_category] = 1.0
        
        # Hedef yaş (rastgele daha yaşlı bir yaş)
        target_ages = [10, 20, 30, 40, 50]  # İstenilen yaş artışları
        target_age_diff = np.random.choice(target_ages)
        target_age = min(99, age + target_age_diff)
        
        target_age_category = min(target_age // 20, 4)
        target_age_vector = torch.zeros(5)
        target_age_vector[target_age_category] = 1.0
        
        return {
            'image': img,
            'age': age,
            'age_vector': age_vector,
            'target_age': target_age,
            'target_age_vector': target_age_vector
        }

# Kayıp fonksiyonları
def generator_loss(fake_out, real_images, fake_images, age_pred, target_age_labels, lambda_id=10.0, lambda_age=1.0):
    # Adversarial loss
    adv_loss = torch.mean((fake_out - 1) ** 2)
    
    # Identity loss (L1)
    identity_loss = torch.mean(torch.abs(fake_images - real_images))
    
    # Age classification loss
    age_loss = nn.CrossEntropyLoss()(age_pred, target_age_labels)
    
    # Toplam kayıp
    total_loss = adv_loss + lambda_id * identity_loss + lambda_age * age_loss
    
    return total_loss, adv_loss, identity_loss, age_loss

def discriminator_loss(real_out, fake_out):
    # Gerçek görüntülerin kaybı
    real_loss = torch.mean((real_out - 1) ** 2)
    
    # Sahte görüntülerin kaybı
    fake_loss = torch.mean(fake_out ** 2)
    
    # Toplam kayıp
    total_loss = real_loss + fake_loss
    
    return total_loss

# Eğitim fonksiyonu
def train(args):
    # Eğitim için cihaz seçimi
    device = torch.device(args.device)
    
    # Veri yükleme
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = FaceAgingDataset(args.data_dir, transform, mode='train')
    val_dataset = FaceAgingDataset(args.data_dir, transform, mode='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Model kurulumu
    generator = Generator(
        conv_dim=args.conv_dim, 
        c_dim=args.c_dim, 
        num_residual_blocks=args.num_residual_blocks
    ).to(device)
    
    discriminator = Discriminator(
        conv_dim=args.conv_dim, 
        c_dim=args.c_dim
    ).to(device)
    
    age_classifier = AgeClassifier(
        conv_dim=args.conv_dim, 
        c_dim=args.c_dim
    ).to(device)
    
    # Optimizasyon
    g_optimizer = optim.Adam(
        generator.parameters(), 
        lr=args.lr_g, 
        betas=(args.beta1, args.beta2)
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(), 
        lr=args.lr_d, 
        betas=(args.beta1, args.beta2)
    )
    
    ac_optimizer = optim.Adam(
        age_classifier.parameters(), 
        lr=args.lr_ac, 
        betas=(args.beta1, args.beta2)
    )
    
    # Eğitim öncesi hazırlık
    start_epoch = 0
    
    # Eğitimi devam ettirme
    if args.resume and os.path.exists(args.resume):
        print(f"Eğitim devam ettiriliyor: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        age_classifier.load_state_dict(checkpoint['age_classifier'])
        
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        ac_optimizer.load_state_dict(checkpoint['ac_optimizer'])
        
        start_epoch = checkpoint['epoch'] + 1
    
    # Çıktı dizinini oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Eğitim döngüsü
    for epoch in range(start_epoch, args.epochs):
        # Eğitim modu
        generator.train()
        discriminator.train()
        age_classifier.train()
        
        # Eğitim için ilerleme çubuğu
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # İstatistikler
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_id_loss = 0
        epoch_age_loss = 0
        
        for batch_idx, batch in enumerate(train_progress):
            # Veriyi hazırlama
            real_images = batch['image'].to(device)
            real_age_vectors = batch['age_vector'].to(device)
            target_age_vectors = batch['target_age_vector'].to(device)
            target_age_labels = torch.argmax(target_age_vectors, dim=1)
            
            batch_size = real_images.size(0)
            
            # ---------------------
            #  Discriminator Eğitimi
            # ---------------------
            d_optimizer.zero_grad()
            
            # Sahte görüntüler oluştur
            fake_images = generator(real_images, target_age_vectors)
            
            # Gerçek ve sahte görüntülerin değerlendirilmesi
            real_out = discriminator(real_images, target_age_vectors)
            fake_out = discriminator(fake_images.detach(), target_age_vectors)
            
            # Discriminator kaybı
            d_loss = discriminator_loss(real_out, fake_out)
            
            # Geriye yayılım
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            #  Age Classifier Eğitimi
            # ---------------------
            ac_optimizer.zero_grad()
            
            # Gerçek ve sahte görüntülerden yaş tahmini
            real_age_pred = age_classifier(real_images)
            fake_age_pred = age_classifier(fake_images.detach())
            
            # Gerçek görüntüler için yaş kaybı
            real_age_labels = torch.argmax(real_age_vectors, dim=1)
            real_age_loss = nn.CrossEntropyLoss()(real_age_pred, real_age_labels)
            
            # Sahte görüntüler için yaş kaybı
            fake_age_loss = nn.CrossEntropyLoss()(fake_age_pred, target_age_labels)
            
            # Toplam yaş kaybı
            age_cls_loss = real_age_loss + fake_age_loss
            
            # Geriye yayılım
            age_cls_loss.backward()
            ac_optimizer.step()
            
            # ---------------------
            #  Generator Eğitimi
            # ---------------------
            g_optimizer.zero_grad()
            
            # Sahte görüntüleri yeniden oluştur (gradient bilgisi için)
            fake_images = generator(real_images, target_age_vectors)
            fake_out = discriminator(fake_images, target_age_vectors)
            
            # Yaş tahmini
            fake_age_pred = age_classifier(fake_images)
            
            # Generator kaybı
            g_loss, g_adv_loss, g_id_loss, g_age_loss = generator_loss(
                fake_out, real_images, fake_images, fake_age_pred, target_age_labels,
                lambda_id=args.lambda_id, lambda_age=args.lambda_age
            )
            
            # Geriye yayılım
            g_loss.backward()
            g_optimizer.step()
            
            # İstatistikleri güncelle
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_id_loss += g_id_loss.item()
            epoch_age_loss += g_age_loss.item()
            
            # İlerleme çubuğunu güncelle
            train_progress.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}',
                'ID_loss': f'{g_id_loss.item():.4f}',
                'Age_loss': f'{g_age_loss.item():.4f}'
            })
            
            # Her n adımda bir örnek görüntüler kaydet
            if batch_idx % args.sample_interval == 0:
                save_sample_images(generator, real_images, target_age_vectors, 
                                 epoch, batch_idx, args.output_dir)
        
        # Epoch ortalamaları
        num_batches = len(train_loader)
        epoch_g_loss /= num_batches
        epoch_d_loss /= num_batches
        epoch_id_loss /= num_batches
        epoch_age_loss /= num_batches
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"G_loss: {epoch_g_loss:.4f}, "
              f"D_loss: {epoch_d_loss:.4f}, "
              f"ID_loss: {epoch_id_loss:.4f}, "
              f"Age_loss: {epoch_age_loss:.4f}")
        
        # Validasyon
        if (epoch + 1) % args.val_interval == 0:
            validate(generator, discriminator, age_classifier, val_loader, device, epoch, args)
        
        # Checkpoint kaydet
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'age_classifier': age_classifier.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'ac_optimizer': ac_optimizer.state_dict()
            }
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Model kaydedildi: {checkpoint_path}")
    
    print("Eğitim tamamlandı!")

# Validasyon fonksiyonu
def validate(generator, discriminator, age_classifier, val_loader, device, epoch, args):
    generator.eval()
    discriminator.eval()
    age_classifier.eval()
    
    val_g_loss = 0
    val_d_loss = 0
    val_id_loss = 0
    val_age_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            real_images = batch['image'].to(device)
            real_age_vectors = batch['age_vector'].to(device)
            target_age_vectors = batch['target_age_vector'].to(device)
            target_age_labels = torch.argmax(target_age_vectors, dim=1)
            
            # Generator çıktısı
            fake_images = generator(real_images, target_age_vectors)
            
            # Discriminator çıktıları
            real_out = discriminator(real_images, target_age_vectors)
            fake_out = discriminator(fake_images, target_age_vectors)
            
            # Yaş tahmini
            fake_age_pred = age_classifier(fake_images)
            
            # Kayıplar
            g_loss, g_adv_loss, g_id_loss, g_age_loss = generator_loss(
                fake_out, real_images, fake_images, fake_age_pred, target_age_labels,
                lambda_id=args.lambda_id, lambda_age=args.lambda_age
            )
            
            d_loss = discriminator_loss(real_out, fake_out)
            
            # İstatistikleri güncelle
            val_g_loss += g_loss.item()
            val_d_loss += d_loss.item()
            val_id_loss += g_id_loss.item()
            val_age_loss += g_age_loss.item()
    
    # Ortalamaları hesapla
    num_batches = len(val_loader)
    val_g_loss /= num_batches
    val_d_loss /= num_batches
    val_id_loss /= num_batches
    val_age_loss /= num_batches
    
    print(f"Validasyon (Epoch {epoch+1}) - "
          f"G_loss: {val_g_loss:.4f}, "
          f"D_loss: {val_d_loss:.4f}, "
          f"ID_loss: {val_id_loss:.4f}, "
          f"Age_loss: {val_age_loss:.4f}")
    
    # Validasyon örneklerini kaydet
    sample_batch = next(iter(val_loader))
    real_images = sample_batch['image'].to(device)
    target_age_vectors = sample_batch['target_age_vector'].to(device)
    
    save_sample_images(generator, real_images, target_age_vectors, 
                     epoch, 0, args.output_dir, prefix='val')

# Örnek görüntüleri kaydetme fonksiyonu
def save_sample_images(generator, real_images, target_age_vectors, epoch, batch_idx, output_dir, prefix='train'):
    # Sadece ilk 8 görüntüyü al
    num_samples = min(8, real_images.size(0))
    real_images = real_images[:num_samples]
    target_age_vectors = target_age_vectors[:num_samples]
    
    # Değerlendirme modunda
    generator.eval()
    
    with torch.no_grad():
        # Yaşlandırılmış görüntüleri oluştur
        fake_images = generator(real_images, target_age_vectors)
    
    # Eğitim moduna geri dön
    generator.train()
    
    # Görsel grid oluşturmak için görüntüleri hazırla
    real_images = real_images.detach().cpu().numpy()
    fake_images = fake_images.detach().cpu().numpy()
    
    # Normalleştirmeyi geri al
    real_images = ((real_images + 1) / 2 * 255).astype(np.uint8)
    fake_images = ((fake_images + 1) / 2 * 255).astype(np.uint8)
    
    # Görüntüleri grid olarak düzenle
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Gerçek görüntüler (üst satır)
        img_real = np.transpose(real_images[i], (1, 2, 0))
        axes[0, i].imshow(img_real)
        axes[0, i].set_title("Gerçek")
        axes[0, i].axis('off')
        
        # Yaşlandırılmış görüntüler (alt satır)
        img_fake = np.transpose(fake_images[i], (1, 2, 0))
        axes[1, i].imshow(img_fake)
        
        # Hedef yaş kategorisini göster
        age_category = torch.argmax(target_age_vectors[i]).item()
        age_ranges = ["0-19", "20-39", "40-59", "60-79", "80+"]
        axes[1, i].set_title(f"Yaş: {age_ranges[age_category]}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Görüntüyü kaydet
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_path = os.path.join(sample_dir, f'{prefix}_epoch_{epoch+1}_batch_{batch_idx}.png')
    plt.savefig(sample_path)
    plt.close()

# Argümanları ayrıştırma fonksiyonu
def parse_args():
    parser = argparse.ArgumentParser(description='Yüz Yaşlandırma GAN Eğitimi')
    
    # Veri ve model yolları
    parser.add_argument('--data_dir', type=str, default='cleandataset',
                        help='Veri seti dizini')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Modellerin ve örneklerin kaydedileceği dizin')
    parser.add_argument('--resume', type=str, default=None,
                        help='Eğitimi devam ettirmek için checkpoint dosyası')
    
    # Eğitim parametreleri
    parser.add_argument('--epochs', type=int, default=100,
                        help='Toplam epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch boyutu')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Generator için öğrenme hızı')
    parser.add_argument('--lr_d', type=float, default=0.0001,
                        help='Discriminator için öğrenme hızı')
    parser.add_argument('--lr_ac', type=float, default=0.0001,
                        help='Age Classifier için öğrenme hızı')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam optimizer için beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam optimizer için beta2')
    parser.add_argument('--lambda_id', type=float, default=10.0,
                        help='Identity loss ağırlığı')
    parser.add_argument('--lambda_age', type=float, default=1.0,
                        help='Age classification loss ağırlığı')
    
    # Model parametreleri
    parser.add_argument('--conv_dim', type=int, default=64,
                        help='İlk konvolüsyon katmanındaki filtre sayısı')
    parser.add_argument('--c_dim', type=int, default=5,
                        help='Yaş vektörü boyutu (kategori sayısı)')
    parser.add_argument('--num_residual_blocks', type=int, default=6,
                        help='Generator için kalıntı (residual) blok sayısı')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Görüntü boyutu')
    
    # Sistem parametreleri
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Veri yükleyici için iş parçacığı sayısı')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Eğitim için kullanılacak cihaz (cuda/cpu)')
    
    # Kaydetme ve değerlendirme
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Kaç epoch\'ta bir model kaydedileceği')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Kaç epoch\'ta bir validasyon yapılacağı')
    parser.add_argument('--sample_interval', type=int, default=100,
                        help='Kaç batch\'te bir örnek görüntüler kaydedileceği')
    
    return parser.parse_args()

# Ana fonksiyon
def main():
    # Argümanları al
    args = parse_args()
    
    # Eğitimi başlat
    train(args)

if __name__ == "__main__":
    main()