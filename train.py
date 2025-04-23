import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


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
    
    def __init__(self, conv_dim=64, c_dim=101, num_residual_blocks=6):
        super(Generator, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, conv_dim, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(True)
        ]
        
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2))
            layers.append(nn.ReLU(True))
            curr_dim = curr_dim * 2
        
        for i in range(num_residual_blocks):
            layers.append(ResidualBlock(curr_dim))
        
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2))
            layers.append(nn.ReLU(True))
            curr_dim = curr_dim // 2
        
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, padding=0, bias=False))
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
        
        self.age_mlp = nn.Sequential(
            nn.Linear(c_dim, curr_dim),
            nn.ReLU(True),
            nn.Linear(curr_dim, curr_dim),
            nn.ReLU(True)
        )

    def forward(self, x, c):
        c_embed = self.age_mlp(c)
        c_embed = c_embed.view(c_embed.size(0), c_embed.size(1), 1, 1)
        c_embed = c_embed.repeat(1, 1, x.size(2), x.size(3))
        
        x = self.main(x)
        
        x = x + 0.1 * c_embed
        
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=101, repeat_num=6):
        super(Discriminator, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src, out_cls

class IPCGAN:
    def __init__(self, model_path=None, device='cpu', image_size=128, c_dim=101):
        self.device = torch.device(device)
        self.image_size = image_size
        self.c_dim = c_dim
        
        self.age_groups = {
            0: 10,
            1: 20,
            2: 40,
        }
        
        self.G = Generator(c_dim=c_dim).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.G.eval()
        
        self.mean = torch.FloatTensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.FloatTensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
    
    def load_model(self, path):
        print(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'generator' in checkpoint:
            self.G.load_state_dict(checkpoint['generator'])
        else:
            self.G.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
    
    def age_to_onehot(self, age_category):
        years_to_add = self.age_groups[age_category]
        
        onehot = torch.zeros(self.c_dim)
        
        if age_category == 0:
            target_age = 30
        elif age_category == 1:
            target_age = 45
        else:
            target_age = 65
        
        target_age = max(1, min(target_age, self.c_dim))
        
        onehot[target_age-1] = 1.0
        
        return onehot
    
    def estimate_age(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        est_age = int(20 + brightness / 10 + contrast / 5)
        
        est_age = max(1, min(est_age, 101))
        
        return est_age
    
    def preprocess_image(self, img):
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        img = img.unsqueeze(0)
        
        return img.to(self.device)
    
    def postprocess_image(self, tensor):
        tensor = tensor.detach().cpu()
        
        tensor = tensor * 0.5 + 0.5
        
        tensor = torch.clamp(tensor, 0, 1)
        
        img = tensor.numpy()[0].transpose(1, 2, 0)
        
        img = (img * 255).astype(np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def age_face(self, img, target_age_category):
        self.G.eval()
        
        with torch.no_grad():
            x = self.preprocess_image(img)
            
            current_age = self.estimate_age(img)
            print(f"Estimated current age: {current_age}")
            
            years_to_add = self.age_groups[target_age_category]
            target_age = min(101, current_age + years_to_add)
            print(f"Target age: {target_age} (adding {years_to_add} years)")
            
            c = torch.zeros(self.c_dim)
            c[target_age-1] = 1.0
            c = c.unsqueeze(0).to(self.device)
            
            aged_face = self.G(x, c)
            
            aged_face = self.postprocess_image(aged_face)
            
            return aged_face
            
class FaceDetector:
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, img, min_neighbors=5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        return faces
    
    def extract_face(self, img, face_rect, padding=0.2):
        x, y, w, h = face_rect
        
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img.shape[1], x + w + pad_w)
        y2 = min(img.shape[0], y + h + pad_h)
        
        face_img = img[y1:y2, x1:x2]
        padded_rect = (x1, y1, x2-x1, y2-y1)
        
        return face_img, padded_rect

def train_model(data_path="cleandataset", save_path="models", epochs=100, batch_size=16):
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    import glob
    import random
    
    os.makedirs(save_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    class FaceAgingDataset(Dataset):
        def __init__(self, data_path, transform=None):
            self.data_path = data_path
            self.transform = transform
            
            self.image_files = []
            self.ages = []
            
            for age_folder in sorted(os.listdir(data_path)):
                if not os.path.isdir(os.path.join(data_path, age_folder)):
                    continue
                
                try:
                    age = int(age_folder)
                except ValueError:
                    continue
                
                folder_path = os.path.join(data_path, age_folder)
                image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                             glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                             glob.glob(os.path.join(folder_path, "*.png"))
                
                self.image_files.extend(image_files)
                self.ages.extend([age] * len(image_files))
            
            print(f"Found {len(self.image_files)} images across {len(set(self.ages))} age groups")
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            age = self.ages[idx]
            
            img = Image.open(img_path).convert("RGB")
            
            if self.transform:
                img = self.transform(img)
            
            age_onehot = torch.zeros(101)
            age_onehot[age-1] = 1
            
            target_age = random.randint(age, 101)
            target_onehot = torch.zeros(101)
            target_onehot[target_age-1] = 1
            
            return img, age_onehot, target_onehot
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = FaceAgingDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    generator = Generator(c_dim=101).to(device)
    discriminator = Discriminator(c_dim=101).to(device)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    adversarial_loss = nn.BCEWithLogitsLoss()
    age_loss = nn.CrossEntropyLoss()
    identity_loss = nn.L1Loss()
    
    print("Starting training...")
    for epoch in range(epochs):
        for i, (imgs, source_ages, target_ages) in enumerate(dataloader):
            imgs = imgs.to(device)
            source_ages = source_ages.to(device)
            target_ages = target_ages.to(device)
            
            real_labels = torch.ones(imgs.size(0), 1, 16, 16).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1, 16, 16).to(device)
            
            d_optimizer.zero_grad()
            
            real_src, real_cls = discriminator(imgs)
            d_real_loss = adversarial_loss(real_src, real_labels)
            d_real_cls_loss = age_loss(real_cls, torch.argmax(source_ages, dim=1))
            
            fake_imgs = generator(imgs, target_ages)
            fake_src, fake_cls = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_src, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss + d_real_cls_loss
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            
            fake_src, fake_cls = discriminator(fake_imgs)
            g_fake_loss = adversarial_loss(fake_src, real_labels)
            g_fake_cls_loss = age_loss(fake_cls, torch.argmax(target_ages, dim=1))
            
            id_loss = identity_loss(fake_imgs, imgs) * 10
            
            g_loss = g_fake_loss + g_fake_cls_loss + id_loss
            g_loss.backward()
            g_optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        if (epoch+1) % 10 == 0:
            model_path = os.path.join(save_path, f"ipcgan_epoch_{epoch+1}.pth")
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch
            }, model_path)
            print(f"Model saved to {model_path}")
    
    final_model_path = os.path.join(save_path, "ipcgan_final.pth")
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IPCGAN Face Aging")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--model", type=str, default="models/ipcgan_final.pth", help="Path to model file")
    parser.add_argument("--image", type=str, default="test_face.jpg", help="Path to test image")
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    
    if args.test:
        gan = IPCGAN(model_path=args.model)
        
        img = cv2.imread(args.image)
        
        if img is None:
            print(f"Error: Could not load image {args.image}")
            sys.exit(1)
        
        detector = FaceDetector()
        
        faces = detector.detect_faces(img)
        
        if len(faces) > 0:
            face_img, face_rect = detector.extract_face(img, faces[0])
            
            cv2.imshow("Original", face_img)
            
            for i in range(3):
                aged_face = gan.age_face(face_img, target_age_category=i)
                cv2.imshow(f"Aged (+{gan.age_groups[i]} years)", aged_face)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No faces detected")