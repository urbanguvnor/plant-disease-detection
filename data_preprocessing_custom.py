"""
Data Preprocessing Module - Custom for train/val/test split dataset
This handles loading data that's already split into train/val/test folders
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class PlantDiseaseDatasetCustom(Dataset):
    """
    Custom Dataset class for datasets with train/val/test structure
    
    Handles folder structure like:
    data/
    ├── plant1/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── plant2/
    │   ├── train/
    │   ├── val/
    │   └── test/
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialize the dataset
        
        Args:
            root_dir: Path to dataset folder (e.g., 'data')
            split: Which split to use ('train', 'val', or 'test')
            transform: Image transformations to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Get all plant folders (e.g., 'apple', 'bell_pepper')
        plant_folders = [d for d in os.listdir(root_dir) 
                        if os.path.isdir(os.path.join(root_dir, d))]
        
        # Build class names from plant folders and their subfolders
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        
        class_idx = 0
        
        for plant in sorted(plant_folders):
            plant_path = os.path.join(root_dir, plant)
            split_path = os.path.join(plant_path, split)
            
            if not os.path.exists(split_path):
                print(f"⚠️  Warning: {split_path} does not exist, skipping...")
                continue
            
            # Get disease/condition folders inside train/val/test
            disease_folders = [d for d in os.listdir(split_path) 
                             if os.path.isdir(os.path.join(split_path, d))]
            
            for disease in sorted(disease_folders):
                # Create class name like "apple_healthy" or "tomato_early_blight"
                class_name = f"{plant}_{disease}"
                self.classes.append(class_name)
                self.class_to_idx[class_name] = class_idx
                
                # Get all images in this class folder
                disease_path = os.path.join(split_path, disease)
                image_files = [f for f in os.listdir(disease_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
                
                for img_name in image_files:
                    img_path = os.path.join(disease_path, img_name)
                    self.samples.append((img_path, class_idx))
                
                class_idx += 1
        
        print(f" Loaded {len(self.samples)} images from {len(self.classes)} classes ({split} split)")
    
    def __len__(self):
        """Return the total number of images"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single image and its label
        
        Args:
            idx: Index of the image to get
            
        Returns:
            image: Preprocessed image tensor
            label: Class label (number)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms(image_size=224):
    """
    Define image transformations for training and validation
    
    Args:
        image_size: Size to resize images to (default: 224x224)
        
    Returns:
        train_transform: Transformations for training data
        val_transform: Transformations for validation/test data
    """
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def prepare_dataloaders(data_dir='data', 
                       batch_size=32, 
                       num_workers=0):
    """
    Prepare data loaders for training, validation, and testing
    
    Args:
        data_dir: Path to dataset (should contain plant folders)
        batch_size: Number of images per batch
        num_workers: Number of CPU threads for loading data (use 0 for Windows)
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        class_names: List of disease class names
    """
    
    print("\n" + "="*60)
    print("PREPARING DATA LOADERS")
    print("="*60)
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Load datasets for each split
    train_dataset = PlantDiseaseDatasetCustom(
        data_dir, 
        split='train', 
        transform=train_transform
    )
    
    val_dataset = PlantDiseaseDatasetCustom(
        data_dir, 
        split='val', 
        transform=val_transform
    )
    
    test_dataset = PlantDiseaseDatasetCustom(
        data_dir, 
        split='test', 
        transform=val_transform
    )
    
    print(f"\n Dataset Split:")
    print(f"   Training:   {len(train_dataset):6d} images")
    print(f"   Validation: {len(val_dataset):6d} images")
    print(f"   Testing:    {len(test_dataset):6d} images")
    print(f"   Total:      {len(train_dataset) + len(val_dataset) + len(test_dataset):6d} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n Data loaders created successfully!")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    # Test the data preprocessing
    print("="*60)
    print("TESTING DATA PREPROCESSING")
    print("="*60)
    
    try:
        train_loader, val_loader, test_loader, classes = prepare_dataloaders()
        
        print(f"\n  Class names ({len(classes)} total):")
        for i, cls in enumerate(classes):
            print(f"   {i:3d}: {cls}")
        
        # Test loading one batch
        print("\n Testing batch loading...")
        images, labels = next(iter(train_loader))
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Sample labels: {labels[:5].tolist()}")
        
        print("\n Data preprocessing test complete!")
        
    except Exception as e:
        print(f"\n Error: {e}")
        print("\nPlease check:")
        print("1. Dataset is in 'data' folder")
        print("2. Each plant has 'train', 'val', 'test' subfolders")
        print("3. Each split folder contains disease class folders")
        print("4. Disease folders contain image files")