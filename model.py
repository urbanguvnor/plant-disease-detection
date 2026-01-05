"""
Model Architecture Module
Defines the CNN model for plant disease classification
"""

import torch
import torch.nn as nn
import torchvision.models as models

class PlantDiseaseClassifier(nn.Module):
    """
    Convolutional Neural Network for Plant Disease Classification
    
    This uses Transfer Learning with a pre-trained ResNet model.
    """
    
    def __init__(self, num_classes, pretrained=True, model_name='resnet50'):
        """
        Initialize the model
        
        Args:
            num_classes: Number of disease classes to predict
            pretrained: Whether to use pre-trained weights (recommended: True)
            model_name: Which pre-trained model to use ('resnet18', 'resnet50')
        """
        super(PlantDiseaseClassifier, self).__init__()
        
        self.model_name = model_name
        
        print(f"\n Building {model_name} model...")
        
        # Load pre-trained model
        if model_name == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                self.backbone = models.resnet18(weights=None)
            num_features = self.backbone.fc.in_features
            # Replace final layer with our custom classifier
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            # Replace final layer
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        else:
            raise ValueError(f"Model {model_name} not supported. Use 'resnet18' or 'resnet50'")
        
        print(f" Model built with {num_classes} output classes")
        if pretrained:
            print("   Using pre-trained weights (Transfer Learning)")
    
    def forward(self, x):
        """
        Forward pass - how data flows through the network
        
        Args:
            x: Input images (batch of images)
            
        Returns:
            Output predictions (logits)
        """
        return self.backbone(x)
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

def create_model(num_classes, model_type='resnet50', pretrained=True):
    """
    Factory function to create models
    
    Args:
        num_classes: Number of disease classes
        model_type: Type of model ('resnet18' or 'resnet50')
        pretrained: Use pre-trained weights
        
    Returns:
        model: PyTorch model
    """
    
    model = PlantDiseaseClassifier(num_classes, pretrained, model_type)
    
    # Count parameters
    total, trainable = model.count_parameters()
    
    print(f"\n Model Statistics:")
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    print("="*60)
    print("MODEL ARCHITECTURE TEST")
    print("="*60)
    
    # Create a test model with 38 classes
    num_classes = 38
    
    # Test ResNet50
    model = create_model(num_classes, model_type='resnet50', pretrained=True)
    
    # Test with dummy input
    print("\n Testing forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n Model test complete!")