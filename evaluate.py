"""
Model Evaluation Script
Evaluates trained model on test set and generates all metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
from tqdm import tqdm
import os
import json

from data_preprocessing_custom import prepare_dataloaders
from model import create_model

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set and return all metrics
    """
    
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)
    
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Evaluate on test set
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Print overall results
    print(f"\n{'='*60}")
    print(f"OVERALL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"{'='*60}")
    
    # Save metrics to file
    metrics = {
        'accuracy': float(accuracy * 100),
        'precision': float(precision * 100),
        'recall': float(recall * 100),
        'f1_score': float(f1 * 100)
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n Metrics saved to: results/test_metrics.json")
    
    # Generate detailed classification report
    print(f"\n{'='*60}")
    print(f"DETAILED CLASSIFICATION REPORT")
    print(f"{'='*60}")
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Save classification report
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n Classification report saved to: results/classification_report.txt")
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'metrics': metrics
    }

def plot_confusion_matrix(labels, predictions, class_names, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Determine figure size based on number of classes
    num_classes = len(class_names)
    fig_size = max(12, num_classes * 0.4)
    
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_normalized, 
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14, pad=20)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Confusion matrix saved to: {save_path}")
    plt.close()

def plot_per_class_accuracy(labels, predictions, class_names, save_path='results/per_class_accuracy.png'):
    """Plot per-class accuracy"""
    
    # Calculate per-class accuracy
    accuracies = []
    for i in range(len(class_names)):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = (predictions[mask] == labels[mask]).mean()
            accuracies.append(class_acc * 100)
        else:
            accuracies.append(0.0)
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    # Plot
    fig_height = max(8, len(class_names) * 0.3)
    plt.figure(figsize=(12, fig_height))
    bars = plt.barh(range(len(sorted_classes)), sorted_accuracies)
    
    # Color bars based on accuracy
    for i, bar in enumerate(bars):
        if sorted_accuracies[i] >= 90:
            bar.set_color('green')
        elif sorted_accuracies[i] >= 70:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.yticks(range(len(sorted_classes)), sorted_classes, fontsize=8)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, pad=20)
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Per-class accuracy plot saved to: {save_path}")
    plt.close()
    
    # Find top and bottom performers
    top_5_idx = sorted_indices[-5:][::-1]
    bottom_5_idx = sorted_indices[:5]
    
    print(f"\n{'='*60}")
    print(f"TOP 5 PERFORMING CLASSES")
    print(f"{'='*60}")
    for idx in top_5_idx:
        print(f"  {class_names[idx]:40s}: {accuracies[idx]:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"BOTTOM 5 PERFORMING CLASSES")
    print(f"{'='*60}")
    for idx in bottom_5_idx:
        print(f"  {class_names[idx]:40s}: {accuracies[idx]:.2f}%")
    
    return accuracies

def analyze_errors(model, test_loader, device, class_names, num_examples=6):
    """Analyze misclassified examples"""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING MISCLASSIFIED EXAMPLES")
    print(f"{'='*60}")
    
    model.eval()
    model = model.to(device)
    
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            mask = preds != labels
            if mask.sum() > 0:
                for i in range(len(inputs)):
                    if mask[i]:
                        misclassified.append({
                            'image': inputs[i].cpu(),
                            'true_label': labels[i].item(),
                            'pred_label': preds[i].item(),
                            'confidence': probs[i].max().item()
                        })
            
            if len(misclassified) >= num_examples:
                break
    
    if len(misclassified) == 0:
        print("    No misclassified examples found!")
        return
    
    print(f"   Found {len(misclassified)} misclassified examples")
    
    # Plot misclassified examples
    num_to_show = min(num_examples, len(misclassified))
    cols = 3
    rows = (num_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_to_show):
        row = idx // cols
        col = idx % cols
        
        example = misclassified[idx]
        
        # Denormalize image
        img = example['image'].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(
            f"True: {class_names[example['true_label']][:25]}\n"
            f"Pred: {class_names[example['pred_label']][:25]}\n"
            f"Conf: {example['confidence']:.2%}",
            fontsize=9
        )
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for idx in range(num_to_show, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path = 'results/misclassified_examples.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Misclassified examples saved to: {save_path}")
    plt.close()

def main():
    """Main evaluation function"""
    
    print("\n" + "="*60)
    print("PLANT DISEASE DETECTION - EVALUATION")
    print("="*60)
    
    # Configuration
    model_path = 'models/best_model.pth'
    data_dir = 'data'
    batch_size = 32
    model_type = 'resnet50'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n Model not found at {model_path}")
        print("   Please train the model first using train.py")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Using device: {device}")
    
    # Load data
    print("\n Loading test data...")
    _, _, test_loader, class_names = prepare_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0
    )
    
    num_classes = len(class_names)
    print(f"   Number of classes: {num_classes}")
    
    # Create model
    print(f"\n Loading model...")
    model = create_model(num_classes, model_type=model_type, pretrained=False)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"   Best validation accuracy: {checkpoint['best_acc']:.4f}")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    plot_confusion_matrix(
        results['labels'], 
        results['predictions'], 
        class_names
    )
    
    accuracies = plot_per_class_accuracy(
        results['labels'],
        results['predictions'],
        class_names
    )
    
    analyze_errors(model, test_loader, device, class_names)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n All results saved in 'results/' folder:")
    print(f"   - test_metrics.json (overall metrics)")
    print(f"   - classification_report.txt (detailed per-class)")
    print(f"   - confusion_matrix.png")
    print(f"   - per_class_accuracy.png")
    print(f"   - misclassified_examples.png")
    
    print(f"\nMetrics Collected:")
    print(f"   Test Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   Test Precision: {results['precision']*100:.2f}%")
    print(f"   Test Recall:    {results['recall']*100:.2f}%")
    print(f"   Test F1-Score:  {results['f1']*100:.2f}%")

if __name__ == "__main__":
    main()