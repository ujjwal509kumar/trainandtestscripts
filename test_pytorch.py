import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("🔍 PYTORCH MODEL TESTING")
print("=" * 50)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# --- Configuration ---
MODEL_PATH = Path('./models_pytorch/pytorch_osteoporosis_model.pth')
TEST_IMAGE_PATH = Path('./test_high_quality')  # Path to test images
IMG_SIZE = 512
CLASS_NAMES = ['normal', 'osteopenia', 'osteoporosis']
NUM_CLASSES = len(CLASS_NAMES)

# --- Model Architecture (same as training) ---
class OptimizedOsteoporosisModel(nn.Module):
    def __init__(self, num_classes=3):
        super(OptimizedOsteoporosisModel, self).__init__()
        
        # Use EfficientNet-B4 as backbone
        self.backbone = models.efficientnet_b4(pretrained=False)  # No pretrained for loading
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
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

# --- Load Model ---
def load_model(model_path):
    """Load trained PyTorch model"""
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return None
    
    # Create model
    model = OptimizedOsteoporosisModel(num_classes=NUM_CLASSES)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    return model

# --- Image Preprocessing ---
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    """Preprocess single image for prediction"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    image_tensor = test_transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device), image

# --- Prediction Functions ---
def predict_single_image(model, image_path):
    """Predict single image"""
    image_tensor, original_image = preprocess_image(image_path)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy(), original_image

def test_model_on_dataset(model, test_path):
    """Test model on entire dataset"""
    print(f"\n🔍 Testing model on dataset: {test_path}")
    
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = test_path / class_name
        if not class_path.exists():
            print(f"⚠️ Class folder not found: {class_name}")
            continue
        
        image_files = list(class_path.glob('*.jpg'))
        print(f"📊 Testing {class_name}: {len(image_files)} images")
        
        class_correct = 0
        for image_file in image_files:
            try:
                predicted_class, confidence, probabilities, _ = predict_single_image(model, image_file)
                
                all_predictions.append(predicted_class)
                all_true_labels.append(class_idx)
                all_confidences.append(confidence)
                
                if predicted_class == class_idx:
                    class_correct += 1
                    
            except Exception as e:
                print(f"❌ Error processing {image_file}: {e}")
        
        class_accuracy = class_correct / len(image_files) if image_files else 0
        print(f"   Accuracy: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    return all_predictions, all_true_labels, all_confidences

# --- Visualization Functions ---
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[name.capitalize() for name in class_names],
                yticklabels=[name.capitalize() for name in class_names])
    plt.title('PyTorch Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('pytorch_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, test_path, num_samples=9):
    """Visualize predictions on sample images"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('PyTorch Model Predictions', fontsize=16)
    
    sample_count = 0
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = test_path / class_name
        if not class_path.exists():
            continue
        
        image_files = list(class_path.glob('*.jpg'))[:3]  # 3 samples per class
        
        for i, image_file in enumerate(image_files):
            if sample_count >= 9:
                break
                
            try:
                predicted_class, confidence, probabilities, original_image = predict_single_image(model, image_file)
                
                row = sample_count // 3
                col = sample_count % 3
                
                axes[row, col].imshow(original_image)
                axes[row, col].set_title(
                    f'True: {class_name.capitalize()}\\n'
                    f'Pred: {CLASS_NAMES[predicted_class].capitalize()}\\n'
                    f'Conf: {confidence:.3f}',
                    fontsize=10
                )
                axes[row, col].axis('off')
                
                sample_count += 1
                
            except Exception as e:
                print(f"❌ Error visualizing {image_file}: {e}")
    
    plt.tight_layout()
    plt.savefig('pytorch_predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- Interactive Testing ---
def interactive_test(model):
    """Interactive testing interface"""
    print(f"\n🎯 INTERACTIVE TESTING MODE")
    print("Enter image path (or 'quit' to exit):")
    
    while True:
        user_input = input("Image path: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        image_path = Path(user_input)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            continue
        
        try:
            predicted_class, confidence, probabilities, original_image = predict_single_image(model, image_path)
            
            print(f"\n📊 PREDICTION RESULTS:")
            print(f"   Predicted Class: {CLASS_NAMES[predicted_class].capitalize()}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"   All Probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"     {CLASS_NAMES[i].capitalize()}: {prob:.4f} ({prob*100:.2f}%)")
            print()
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")

# --- Main Testing Function ---
def main():
    """Main testing function"""
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Test on dataset
    if TEST_IMAGE_PATH.exists():
        predictions, true_labels, confidences = test_model_on_dataset(model, TEST_IMAGE_PATH)
        
        if predictions:
            # Calculate overall accuracy
            overall_accuracy = np.mean(np.array(predictions) == np.array(true_labels))
            print(f"\n🎯 OVERALL RESULTS:")
            print(f"   Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
            print(f"   Average Confidence: {np.mean(confidences):.4f}")
            
            # Classification report
            print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
            class_names_cap = [name.capitalize() for name in CLASS_NAMES]
            print(classification_report(true_labels, predictions, target_names=class_names_cap))
            
            # Plot confusion matrix
            plot_confusion_matrix(true_labels, predictions, CLASS_NAMES)
            
            # Visualize predictions
            visualize_predictions(model, TEST_IMAGE_PATH)
    
    # Interactive testing
    interactive_test(model)

if __name__ == "__main__":
    main()