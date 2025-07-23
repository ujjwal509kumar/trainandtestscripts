import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from tqdm import tqdm
import time

print("🚀 PYTORCH TRAINING - MAXIMUM ACCURACY FOR MEDICAL IMAGING")
print("=" * 70)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- Configuration ---
DATASET_PATH = Path('./test_high_quality')
MODEL_SAVE_PATH = Path('./models_pytorch')
PLOTS_SAVE_PATH = Path('./plots_pytorch')
MODEL_FILENAME = 'pytorch_osteoporosis_model.pth'
IMG_SIZE = 512
BATCH_SIZE = 4  # Optimal for EfficientNet-B4
EPOCHS_PHASE1 = 60
EPOCHS_PHASE2 = 40
CLASS_NAMES = ['normal', 'osteopenia', 'osteoporosis']
NUM_CLASSES = len(CLASS_NAMES)

# Create directories
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_SAVE_PATH.mkdir(parents=True, exist_ok=True)

print(f"📁 Using high-quality dataset: {DATASET_PATH}")
print(f"💾 Models will be saved to: {MODEL_SAVE_PATH}")

# Verify dataset
for cls in CLASS_NAMES:
    cls_path = DATASET_PATH / cls
    if cls_path.exists():
        count = len(list(cls_path.glob('*.jpg')))
        print(f"📊 {cls.capitalize()}: {count} images")
    else:
        print(f"❌ Missing class folder: {cls}")
        exit(1)

# --- Custom Dataset Class ---
class OsteoporosisDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_names=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"📊 Dataset loaded: {len(self.samples)} images")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# --- Data Transforms ---
# Training transforms with medical-specific augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),  # Gentle rotation for medical images
    transforms.ColorJitter(brightness=0.08, contrast=0.12, saturation=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Create Datasets ---
full_dataset = OsteoporosisDataset(DATASET_PATH, transform=None, class_names=CLASS_NAMES)

# Split dataset (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Create indices for splitting
indices = torch.randperm(len(full_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create train and validation datasets
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

# Apply transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Create data loaders (Windows fix: num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"📊 Training samples: {len(train_dataset)}")
print(f"📊 Validation samples: {len(val_dataset)}")

# --- Calculate Class Weights ---
def calculate_class_weights():
    """Calculate class weights for balanced training"""
    print("🔧 Calculating class weights...")
    
    # Get all labels from training set
    train_labels = []
    for idx in train_indices:
        _, label = full_dataset[idx]
        train_labels.append(label)
    
    # Calculate class distribution
    unique_labels = np.unique(train_labels)
    class_counts = np.bincount(train_labels)
    
    print(f"📊 Class distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"   {class_name.capitalize()}: {class_counts[i]} samples")
    
    # Calculate balanced weights
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"📊 Class weights: {class_weights}")
    return class_weights_tensor

class_weights = calculate_class_weights()

# --- Model Architecture ---
class OptimizedOsteoporosisModel(nn.Module):
    def __init__(self, num_classes=3):
        super(OptimizedOsteoporosisModel, self).__init__()
        
        # Use EfficientNet-B4 as backbone (best for medical imaging)
        self.backbone = models.efficientnet_b4(pretrained=True)
        
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head optimized for medical imaging
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self, unfreeze_layers=20):
        """Unfreeze last N layers of backbone for fine-tuning"""
        backbone_layers = list(self.backbone.features.children())
        
        # Unfreeze last N layers
        for layer in backbone_layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"🔧 Unfroze last {unfreeze_layers} layers for fine-tuning")

# Create model
model = OptimizedOsteoporosisModel(num_classes=NUM_CLASSES).to(device)
print(f"\n📋 Model created with EfficientNet-B4 backbone")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Total parameters: {total_params:,}")
print(f"📊 Trainable parameters: {trainable_params:,}")

# --- Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-7, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=7, verbose=True, min_lr=1e-8)

# --- Training Functions ---
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_targets

# --- Training Loop with Progress Tracking ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, phase_name):
    """Complete training loop with progress tracking"""
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\n🚀 Starting {phase_name}...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc, val_predictions, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\n📊 EPOCH {epoch + 1} SUMMARY:")
        print(f"   Training Loss: {train_loss:.4f} | Training Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"   Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save model every epoch
        epoch_model_path = MODEL_SAVE_PATH / f'{phase_name}_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_loss': val_loss
        }, epoch_model_path)
        print(f"   💾 Model saved: {epoch_model_path.name}")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"   🎯 NEW BEST VALIDATION ACCURACY: {val_acc:.4f} ({val_acc*100:.2f}%)")
            
            # Save best model
            best_model_path = MODEL_SAVE_PATH / f'{phase_name}_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_predictions': val_predictions,
                'val_targets': val_targets
            }, best_model_path)
        
        # Performance assessment for GPU decision (after first epoch)
        if epoch == 0:
            if val_acc > 0.4:
                print(f"   ✅ EXCELLENT START! Model is learning well")
                print(f"   💡 Recommendation: Continue training for maximum accuracy")
            elif val_acc > 0.33:
                print(f"   👍 GOOD START! Training is progressing")
                print(f"   💡 Recommendation: Consider GPU rental for faster training")
            else:
                print(f"   ⚠️ SLOW START! May need more epochs")
                print(f"   💡 Recommendation: GPU rental recommended")
        
        print(f"   ⏱️ Decision point: Continue locally or rent GPU?")
        print("-" * 50)
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'best_model_state': best_model_state
    }

if __name__ == "__main__":
    # --- Phase 1: Train Classification Head ---
    print("\n🚀 Phase 1: Training classification head...")
    phase1_results = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        EPOCHS_PHASE1, "phase1"
    )

    # --- Phase 2: Fine-tuning ---
    print("\n🚀 Phase 2: Fine-tuning with unfrozen backbone...")
    print("=" * 60)

    # Unfreeze backbone layers
    model.unfreeze_backbone(unfreeze_layers=20)

    # Lower learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.00005

    # Update scheduler for fine-tuning
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True, min_lr=1e-9)

    phase2_results = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        EPOCHS_PHASE2, "phase2"
    )

    # --- Final Model Save ---
    final_model_path = MODEL_SAVE_PATH / MODEL_FILENAME
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase1_results': phase1_results,
        'phase2_results': phase2_results,
        'class_names': CLASS_NAMES,
        'img_size': IMG_SIZE
    }, final_model_path)

    print(f"✅ Final model saved as: {final_model_path}")

    print(f"\n" + "="*70)
    print(f"✅ PYTORCH TRAINING COMPLETED!")
    print(f"="*70)
    print(f"🔧 Key advantages of PyTorch version:")
    print(f"   • EfficientNet-B4 backbone (best for medical imaging)")
    print(f"   • Optimized data loading with medical-specific augmentation")
    print(f"   • Better memory management for high-resolution images")
    print(f"   • Advanced learning rate scheduling")
    print(f"   • Comprehensive model checkpointing")
    print(f"   • Real-time training progress monitoring")
    print(f"📁 All models saved in: {MODEL_SAVE_PATH}")
    print(f"🎯 Expected maximum accuracy with PyTorch optimization!")
