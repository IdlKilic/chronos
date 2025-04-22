import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

# IPCGAN Model Classes
class ResidualBlock(nn.Module):
    """Residual Block for Generator"""
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
    """Generator for IPCGAN"""
    def __init__(self, conv_dim=64, c_dim=101, num_residual_blocks=6):
        super(Generator, self).__init__()
        
        # Initial convolutional layers
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, conv_dim, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(True)
        ]
        
        # Downsampling layers
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2))
            layers.append(nn.ReLU(True))
            curr_dim = curr_dim * 2
        
        # Bottleneck layers with residual blocks
        for i in range(num_residual_blocks):
            layers.append(ResidualBlock(curr_dim))
        
        # Upsampling layers
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2))
            layers.append(nn.ReLU(True))
            curr_dim = curr_dim // 2
        
        # Output layer
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, padding=0, bias=False))
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
        
        # Age condition embedding layers
        self.age_mlp = nn.Sequential(
            nn.Linear(c_dim, curr_dim),
            nn.ReLU(True),
            nn.Linear(curr_dim, curr_dim),
            nn.ReLU(True)
        )

    def forward(self, x, c):
        # Embed condition
        c_embed = self.age_mlp(c)
        c_embed = c_embed.view(c_embed.size(0), c_embed.size(1), 1, 1)
        c_embed = c_embed.repeat(1, 1, x.size(2), x.size(3))
        
        # Process input
        x = self.main(x)
        
        # Add condition embedding
        x = x + 0.1 * c_embed
        
        return x

class Discriminator(nn.Module):
    """Discriminator for IPCGAN"""
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
        
        # PatchGAN - classify if each patch is real or fake
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Auxiliary classifier for age
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src, out_cls

class IPCGAN:
    """IPCGAN Face Aging Model"""
    def __init__(self, model_path=None, device='cpu', image_size=128, c_dim=101):
        self.device = torch.device(device)
        self.image_size = image_size
        self.c_dim = c_dim  # Number of age categories (1-101 yaş)
        
        # Age category definitions - for the model
        self.age_groups = {
            0: 10,  # 10 yaş daha yaşlı
            1: 20,  # 20 yaş daha yaşlı
            2: 40,  # 40 yaş daha yaşlı
        }
        
        # Initialize models
        self.G = Generator(c_dim=c_dim).to(self.device)
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Set model to evaluation mode
        self.G.eval()
        
        # Mean and std for normalization
        self.mean = torch.FloatTensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.FloatTensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
    
    def load_model(self, path):
        """Load pre-trained model"""
        print(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'generator' in checkpoint:
            self.G.load_state_dict(checkpoint['generator'])
        else:
            self.G.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
    
    def age_to_onehot(self, age_category):
        """Convert age to one-hot encoding
        
        Args:
            age_category (int): Age category index from combo box
                0: 10 years older
                1: 20 years older
                2: 40 years older
        
        Returns:
            torch.Tensor: One-hot encoding of target age (101 dimensions)
        """
        # Get target age based on age category
        years_to_add = self.age_groups[age_category]
        
        # For demonstration, we'll just use fixed target age vectors
        # In a real application, we would estimate current age and add years_to_add
        
        # Create one-hot encoding (101 dimensions for ages 1-101)
        onehot = torch.zeros(self.c_dim)
        
        # Target ages based on selection (just examples)
        if age_category == 0:  # 10 yaş yaşlandır
            target_age = 30  # Örnek olarak 30 yaşında gösterme
        elif age_category == 1:  # 20 yaş yaşlandır
            target_age = 45  # Örnek olarak 45 yaşında gösterme
        else:  # 40 yaş yaşlandır
            target_age = 65  # Örnek olarak 65 yaşında gösterme
        
        # Ensure target age is within bounds
        target_age = max(1, min(target_age, self.c_dim))
        
        # Set the one-hot encoding (shift by 1 since age starts at 1)
        onehot[target_age-1] = 1.0
        
        return onehot
    
    def estimate_age(self, face_img):
        """Estimate the age of a face.
        
        This is a placeholder function. In a real application, you would use a
        dedicated age estimation model or use the IPCGAN's discriminator.
        
        Args:
            face_img (numpy.ndarray): Face image in BGR format
            
        Returns:
            int: Estimated age (between 1 and 101)
        """
        # This is a placeholder implementation
        # In a real application, you would use a ML model for age estimation
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate some simple features (example only)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Crude age estimate based on image features
        # This is just a placeholder and will not produce accurate results
        # It should be replaced with a proper age estimation model
        est_age = int(20 + brightness / 10 + contrast / 5)
        
        # Ensure age is within bounds
        est_age = max(1, min(est_age, 101))
        
        return est_age
    
    def preprocess_image(self, img):
        """Preprocess image for the model"""
        # Resize to model input size
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Convert from BGR to RGB (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # Convert to tensor and change dimensions to [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img.to(self.device)
    
    def postprocess_image(self, tensor):
        """Convert tensor to numpy image"""
        # Detach from computation graph and move to CPU
        tensor = tensor.detach().cpu()
        
        # Denormalize
        tensor = tensor * 0.5 + 0.5
        
        # Clamp values to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and change dimensions to [H, W, C]
        img = tensor.numpy()[0].transpose(1, 2, 0)
        
        # Convert to uint8
        img = (img * 255).astype(np.uint8)
        
        # Convert from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def age_face(self, img, target_age_category):
        """Age a face image
        
        Args:
            img (numpy.ndarray): Input face image (BGR format)
            target_age_category (int): Age category index (0-2)
                0: 10 years older
                1: 20 years older
                2: 40 years older
        
        Returns:
            numpy.ndarray: Aged face image (BGR format)
        """
        # Ensure model is in evaluation mode
        self.G.eval()
        
        with torch.no_grad():
            # Preprocess image
            x = self.preprocess_image(img)
            
            # Estimate current age
            current_age = self.estimate_age(img)
            print(f"Estimated current age: {current_age}")
            
            # Compute target age
            years_to_add = self.age_groups[target_age_category]
            target_age = min(101, current_age + years_to_add)
            print(f"Target age: {target_age} (adding {years_to_add} years)")
            
            # Prepare age condition
            c = torch.zeros(self.c_dim)
            c[target_age-1] = 1.0  # Adjust index (ages are 1-101, indices are 0-100)
            c = c.unsqueeze(0).to(self.device)
            
            # Generate aged face
            aged_face = self.G(x, c)
            
            # Postprocess image
            aged_face = self.postprocess_image(aged_face)
            
            return aged_face
            
# Simple face detector class using OpenCV's Haar Cascade
class FaceDetector:
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            # Use OpenCV's default face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, img, min_neighbors=5):
        """Detect faces in an image
        
        Args:
            img (numpy.ndarray): Input image (BGR format)
            min_neighbors (int): Minimum number of neighbors for detection
        
        Returns:
            list: List of (x, y, w, h) tuples for each detected face
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        return faces
    
    def extract_face(self, img, face_rect, padding=0.2):
        """Extract a face from an image with padding
        
        Args:
            img (numpy.ndarray): Input image (BGR format)
            face_rect (tuple): Face rectangle (x, y, w, h)
            padding (float): Padding factor for the face rectangle
        
        Returns:
            numpy.ndarray: Extracted face image (BGR format)
            tuple: Original face rectangle with padding
        """
        x, y, w, h = face_rect
        
        # Calculate padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate new rectangle with padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img.shape[1], x + w + pad_w)
        y2 = min(img.shape[0], y + h + pad_h)
        
        # Extract face
        face_img = img[y1:y2, x1:x2]
        padded_rect = (x1, y1, x2-x1, y2-y1)
        
        return face_img, padded_rect

def train_model(data_path="cleandataset", save_path="models", epochs=100, batch_size=16):
    """Train IPCGAN model on face dataset
    
    This is a simplified training function for the IPCGAN model.
    In a real application, this would be much more complex and require
    more sophisticated data loading, augmentation, and training logic.
    
    Args:
        data_path (str): Path to the dataset directory
        save_path (str): Path to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    import glob
    import random
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define dataset
    class FaceAgingDataset(Dataset):
        def __init__(self, data_path, transform=None):
            self.data_path = data_path
            self.transform = transform
            
            # Get all image files
            self.image_files = []
            self.ages = []
            
            # Assuming folder structure: data_path/age_folder/image_files
            for age_folder in sorted(os.listdir(data_path)):
                if not os.path.isdir(os.path.join(data_path, age_folder)):
                    continue
                
                # Extract age from folder name (e.g., "001" -> 1)
                try:
                    age = int(age_folder)
                except ValueError:
                    continue
                
                # Get image files in this age folder
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
            
            # Load image
            img = Image.open(img_path).convert("RGB")
            
            # Apply transformations
            if self.transform:
                img = self.transform(img)
            
            # Create one-hot encoding for age
            age_onehot = torch.zeros(101)  # Ages 1-101
            age_onehot[age-1] = 1  # Adjust index (ages are 1-101, indices are 0-100)
            
            # Pick a random target age that is older
            target_age = random.randint(age, 101)
            target_onehot = torch.zeros(101)
            target_onehot[target_age-1] = 1
            
            return img, age_onehot, target_onehot
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    dataset = FaceAgingDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize models
    generator = Generator(c_dim=101).to(device)
    discriminator = Discriminator(c_dim=101).to(device)
    
    # Define optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Define loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    age_loss = nn.CrossEntropyLoss()
    identity_loss = nn.L1Loss()
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        for i, (imgs, source_ages, target_ages) in enumerate(dataloader):
            imgs = imgs.to(device)
            source_ages = source_ages.to(device)
            target_ages = target_ages.to(device)
            
            # Create labels
            real_labels = torch.ones(imgs.size(0), 1, 16, 16).to(device)  # For PatchGAN
            fake_labels = torch.zeros(imgs.size(0), 1, 16, 16).to(device)
            
            # ------------------
            # Train Discriminator
            # ------------------
            d_optimizer.zero_grad()
            
            # Real images
            real_src, real_cls = discriminator(imgs)
            d_real_loss = adversarial_loss(real_src, real_labels)
            d_real_cls_loss = age_loss(real_cls, torch.argmax(source_ages, dim=1))
            
            # Fake images
            fake_imgs = generator(imgs, target_ages)
            fake_src, fake_cls = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_src, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + d_real_cls_loss
            d_loss.backward()
            d_optimizer.step()
            
            # ------------------
            # Train Generator
            # ------------------
            g_optimizer.zero_grad()
            
            # Adversarial loss
            fake_src, fake_cls = discriminator(fake_imgs)
            g_fake_loss = adversarial_loss(fake_src, real_labels)
            g_fake_cls_loss = age_loss(fake_cls, torch.argmax(target_ages, dim=1))
            
            # Identity loss
            id_loss = identity_loss(fake_imgs, imgs) * 10  # Weight identity loss
            
            # Total generator loss
            g_loss = g_fake_loss + g_fake_cls_loss + id_loss
            g_loss.backward()
            g_optimizer.step()
            
            # Print progress
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        # Save model checkpoints
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
    
    # Save final model
    final_model_path = os.path.join(save_path, "ipcgan_final.pth")
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

# Example usage
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
        # Initialize the model
        gan = IPCGAN(model_path=args.model)
        
        # Load a test image
        img = cv2.imread(args.image)
        
        if img is None:
            print(f"Error: Could not load image {args.image}")
            sys.exit(1)
        
        # Initialize face detector
        detector = FaceDetector()
        
        # Detect faces
        faces = detector.detect_faces(img)
        
        if len(faces) > 0:
            # Extract the first face
            face_img, face_rect = detector.extract_face(img, faces[0])
            
            # Show original face
            cv2.imshow("Original", face_img)
            
            # Age the face for different target ages
            for i in range(3):
                aged_face = gan.age_face(face_img, target_age_category=i)
                cv2.imshow(f"Aged (+{gan.age_groups[i]} years)", aged_face)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No faces detected")