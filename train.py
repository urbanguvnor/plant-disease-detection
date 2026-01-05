"""
Training Script
Trains the plant disease classification model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from data_preprocessing_custom import prepare_dataloaders
from model import create_model

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, device, save_dir='models'):
    """
    Train the model
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save model checkpoints
        
    Returns:
        model: Trained model
        history: Training history
    """
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("="*60 + "\n")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Track best model
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar
            pbar = tqdm(dataloader, desc=f'{phase.capitalize()}')
            
            # Iterate over batches
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
            
            # Save to history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Update learning rate
                scheduler.step(epoch_loss)
                
                # Save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # Save checkpoint
                    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'history': history
                    }, checkpoint_path)
                    print(f' Saved best model (Val Acc: {best_acc:.4f})')
    
    # Training complete
    time_elapsed = time.time() - start_time
    print('\n' + "="*60)
    print('TRAINING COMPLETE')
    print("="*60)
    print(f'Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training and validation metrics"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n Training history plot saved to: {save_path}')
    plt.close()

def main():
    """Main training function"""
    
    # Configuration
    CONFIG = {
        'data_dir': 'data',
        'batch_size': 32,
        'num_epochs': 15,
        'learning_rate': 0.001,
        'model_type': 'resnet50',  # or 'resnet18'
        'pretrained': True,
        'save_dir': 'models'
    }
    
    print("\n" + "="*60)
    print("PLANT DISEASE DETECTION - TRAINING")
    print("="*60)
    print("\n Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        print(f"\n🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n  No GPU detected, using CPU (training will be slower)")
    
    # Prepare data
    print("\n Loading data...")
    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=0  # Set to 0 for Windows
    )
    
    num_classes = len(class_names)
    print(f"\n  Number of classes: {num_classes}")
    
    # Create model
    model = create_model(
        num_classes=num_classes,
        model_type=CONFIG['model_type'],
        pretrained=CONFIG['pretrained']
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=CONFIG['num_epochs'],
        device=device,
        save_dir=CONFIG['save_dir']
    )
    
    # Plot training history
    plot_training_history(history)
    
    print("\n Training pipeline complete!")
    print(f"   Best model saved in: {CONFIG['save_dir']}/best_model.pth")
    print(f"   Training history plot: results/training_history.png")

if __name__ == "__main__":
    main()