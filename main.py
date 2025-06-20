import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ConceptBottleneckModel(nn.Module):
    """
    Concept Bottleneck Model that predicts interpretable concepts
    before making final predictions.
    """
    def __init__(self, 
                 backbone: str = 'resnet18',
                 num_concepts: int = 10,
                 num_classes: int = 10,
                 concept_dim: int = 128,
                 bottleneck_activation: str = 'sigmoid'):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.bottleneck_activation = bottleneck_activation
        
        # Feature extractor (backbone)
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final classifier
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Concept prediction head
        self.concept_head = nn.Sequential(
            nn.Linear(backbone_dim, concept_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(concept_dim, num_concepts)
        )
        
        # Final classification head (uses concepts)
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, num_classes)
        )
        
        # Store concept names for interpretability
        self.concept_names = [f"Concept_{i}" for i in range(num_concepts)]
        
    def forward(self, x: torch.Tensor, return_concepts: bool = False):
        # Extract features
        features = self.backbone(x)
        
        # Predict concepts
        concept_logits = self.concept_head(features)
        
        # Apply bottleneck activation
        if self.bottleneck_activation == 'sigmoid':
            concepts = torch.sigmoid(concept_logits)
        elif self.bottleneck_activation == 'tanh':
            concepts = torch.tanh(concept_logits)
        else:
            concepts = concept_logits
        
        # Final classification using concepts
        class_logits = self.classifier(concepts)
        
        if return_concepts:
            return class_logits, concepts, concept_logits
        return class_logits
    
    def set_concept_names(self, names: List[str]):
        """Set human-readable names for concepts."""
        assert len(names) == self.num_concepts
        self.concept_names = names

class ConceptInterventionModel(ConceptBottleneckModel):
    """
    Extended CBM that allows concept intervention during inference.
    """
    def forward(self, x: torch.Tensor, 
                concept_intervention: Optional[torch.Tensor] = None,
                return_concepts: bool = False):
        # Extract features
        features = self.backbone(x)
        
        # Predict concepts
        concept_logits = self.concept_head(features)
        
        # Apply bottleneck activation
        if self.bottleneck_activation == 'sigmoid':
            concepts = torch.sigmoid(concept_logits)
        elif self.bottleneck_activation == 'tanh':
            concepts = torch.tanh(concept_logits)
        else:
            concepts = concept_logits
        
        # Apply concept intervention if provided
        if concept_intervention is not None:
            # Mask for which concepts to intervene on
            intervention_mask = ~torch.isnan(concept_intervention)
            concepts = torch.where(intervention_mask, concept_intervention, concepts)
        
        # Final classification using concepts
        class_logits = self.classifier(concepts)
        
        if return_concepts:
            return class_logits, concepts, concept_logits
        return class_logits

class CBMTrainer:
    """Trainer class for Concept Bottleneck Models."""
    
    def __init__(self, model: ConceptBottleneckModel, 
                 concept_weight: float = 1.0,
                 class_weight: float = 1.0):
        self.model = model
        self.concept_weight = concept_weight
        self.class_weight = class_weight
        
        self.concept_criterion = nn.BCEWithLogitsLoss()
        self.class_criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.concept_accuracies = []
        self.class_accuracies = []
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        concept_correct = 0
        class_correct = 0
        total_samples = 0
        
        for batch_idx, (data, concepts, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            class_logits, pred_concepts, concept_logits = self.model(
                data, return_concepts=True)
            
            # Compute losses
            concept_loss = self.concept_criterion(concept_logits, concepts.float())
            class_loss = self.class_criterion(class_logits, labels)
            
            total_loss_batch = (self.concept_weight * concept_loss + 
                              self.class_weight * class_loss)
            
            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Calculate accuracies
            concept_pred = (torch.sigmoid(concept_logits) > 0.5).float()
            concept_correct += (concept_pred == concepts).float().sum().item()
            
            class_pred = class_logits.argmax(dim=1)
            class_correct += (class_pred == labels).sum().item()
            
            total_samples += data.size(0)
        
        avg_loss = total_loss / len(dataloader)
        concept_acc = concept_correct / (total_samples * self.model.num_concepts)
        class_acc = class_correct / total_samples
        
        return avg_loss, concept_acc, class_acc
    
    def validate(self, dataloader: DataLoader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        concept_correct = 0
        class_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, concepts, labels in dataloader:
                class_logits, pred_concepts, concept_logits = self.model(
                    data, return_concepts=True)
                
                # Compute losses
                concept_loss = self.concept_criterion(concept_logits, concepts.float())
                class_loss = self.class_criterion(class_logits, labels)
                
                total_loss += (self.concept_weight * concept_loss + 
                             self.class_weight * class_loss).item()
                
                # Calculate accuracies
                concept_pred = (torch.sigmoid(concept_logits) > 0.5).float()
                concept_correct += (concept_pred == concepts).float().sum().item()
                
                class_pred = class_logits.argmax(dim=1)
                class_correct += (class_pred == labels).sum().item()
                
                total_samples += data.size(0)
        
        avg_loss = total_loss / len(dataloader)
        concept_acc = concept_correct / (total_samples * self.model.num_concepts)
        class_acc = class_correct / total_samples
        
        return avg_loss, concept_acc, class_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, lr: float = 0.001):
        """Full training loop."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_concept_acc, train_class_acc = self.train_epoch(
                train_loader, optimizer)
            
            # Validation
            val_loss, val_concept_acc, val_class_acc = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.concept_accuracies.append(val_concept_acc)
            self.class_accuracies.append(val_class_acc)
            
            scheduler.step()
            
            # Print progress
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Concept Acc: {val_concept_acc:.4f}, "
                      f"Class Acc: {val_class_acc:.4f}")
            
            # Save best model
            if val_class_acc > best_val_acc:
                best_val_acc = val_class_acc
                torch.save(self.model.state_dict(), 'best_cbm_model.pth')

class CBMAnalyzer:
    """Analyzer for interpretability and performance trade-offs."""
    
    def __init__(self, model: ConceptBottleneckModel):
        self.model = model
        
    def analyze_concept_importance(self, dataloader: DataLoader):
        """Analyze which concepts are most important for classification."""
        self.model.eval()
        
        concept_activations = []
        class_predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, concepts, labels in dataloader:
                class_logits, pred_concepts, _ = self.model(data, return_concepts=True)
                
                concept_activations.append(pred_concepts.cpu().numpy())
                class_predictions.append(class_logits.argmax(dim=1).cpu().numpy())
                true_labels.append(labels.cpu().numpy())
        
        concept_activations = np.vstack(concept_activations)
        class_predictions = np.concatenate(class_predictions)
        true_labels = np.concatenate(true_labels)
        
        # Calculate concept importance
        importance_scores = {}
        for class_idx in range(self.model.num_classes):
            class_mask = true_labels == class_idx
            if class_mask.sum() > 0:
                class_concepts = concept_activations[class_mask]
                importance_scores[f'Class_{class_idx}'] = class_concepts.mean(axis=0)
        
        return importance_scores
    
    def concept_intervention_analysis(self, dataloader: DataLoader, 
                                    concept_idx: int, intervention_values: List[float]):
        """Analyze the effect of intervening on specific concepts."""
        if not isinstance(self.model, ConceptInterventionModel):
            print("Model doesn't support concept intervention")
            return None
        
        self.model.eval()
        results = {}
        
        for intervention_value in intervention_values:
            class_predictions = []
            
            with torch.no_grad():
                for data, concepts, labels in dataloader:
                    batch_size = data.size(0)
                    
                    # Create intervention tensor
                    intervention = torch.full((batch_size, self.model.num_concepts), 
                                            float('nan'))
                    intervention[:, concept_idx] = intervention_value
                    
                    class_logits = self.model(data, concept_intervention=intervention)
                    class_predictions.extend(class_logits.argmax(dim=1).cpu().numpy())
            
            results[intervention_value] = np.array(class_predictions)
        
        return results
    
    def visualize_concept_space(self, dataloader: DataLoader, max_samples: int = 1000):
        """Visualize the learned concept space using t-SNE."""
        self.model.eval()
        
        concept_activations = []
        true_labels = []
        sample_count = 0
        
        with torch.no_grad():
            for data, concepts, labels in dataloader:
                if sample_count >= max_samples:
                    break
                
                _, pred_concepts, _ = self.model(data, return_concepts=True)
                
                concept_activations.append(pred_concepts.cpu().numpy())
                true_labels.append(labels.cpu().numpy())
                sample_count += data.size(0)
        
        concept_activations = np.vstack(concept_activations)[:max_samples]
        true_labels = np.concatenate(true_labels)[:max_samples]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        concept_2d = tsne.fit_transform(concept_activations)
        
        return concept_2d, true_labels
    
    def compute_interpretability_metrics(self, dataloader: DataLoader):
        """Compute various interpretability metrics."""
        self.model.eval()
        
        concept_activations = []
        concept_logits_list = []
        
        with torch.no_grad():
            for data, concepts, labels in dataloader:
                _, pred_concepts, concept_logits = self.model(data, return_concepts=True)
                
                concept_activations.append(pred_concepts.cpu().numpy())
                concept_logits_list.append(concept_logits.cpu().numpy())
        
        concept_activations = np.vstack(concept_activations)
        concept_logits = np.vstack(concept_logits_list)
        
        # Concept activation statistics
        concept_stats = {
            'mean_activation': concept_activations.mean(axis=0),
            'std_activation': concept_activations.std(axis=0),
            'concept_correlation': np.corrcoef(concept_activations.T),
        }
        
        # Concept confidence (how certain the model is about each concept)
        concept_confidence = np.abs(concept_logits).mean(axis=0)
        concept_stats['confidence'] = concept_confidence
        
        return concept_stats

# Synthetic dataset for demonstration
class SyntheticConceptDataset(Dataset):
    """Synthetic dataset with explicit concepts for demonstration."""
    
    def __init__(self, num_samples: int = 1000, num_concepts: int = 10, 
                 num_classes: int = 3, image_size: int = 64):
        self.num_samples = num_samples
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate synthetic data
        np.random.seed(42)
        
        # Generate concept labels (binary)
        self.concepts = np.random.binomial(1, 0.5, (num_samples, num_concepts))
        
        # Generate class labels based on concepts (with some logic)
        class_probs = np.zeros((num_samples, num_classes))
        for i in range(num_samples):
            # Simple rule: different concept combinations lead to different classes
            concept_sum = self.concepts[i].sum()
            if concept_sum <= 3:
                class_probs[i] = [0.7, 0.2, 0.1]
            elif concept_sum <= 6:
                class_probs[i] = [0.2, 0.6, 0.2]
            else:
                class_probs[i] = [0.1, 0.2, 0.7]
        
        self.labels = np.array([np.random.choice(num_classes, p=prob) 
                               for prob in class_probs])
        
        # Generate synthetic images based on concepts
        self.images = self._generate_images()
        
    def _generate_images(self):
        """Generate synthetic images that reflect the concept structure."""
        images = []
        
        for i in range(self.num_samples):
            # Create base image
            img = np.random.randn(3, self.image_size, self.image_size) * 0.1
            
            # Add patterns based on concepts
            for c in range(self.num_concepts):
                if self.concepts[i, c] == 1:
                    # Add specific pattern for this concept
                    pattern = self._generate_concept_pattern(c)
                    img += pattern * 0.5
            
            # Add some noise
            img += np.random.randn(3, self.image_size, self.image_size) * 0.1
            
            # Normalize
            img = np.clip(img, -2, 2)
            images.append(img.astype(np.float32))
        
        return np.array(images)
    
    def _generate_concept_pattern(self, concept_idx: int):
        """Generate a specific pattern for a concept."""
        pattern = np.zeros((3, self.image_size, self.image_size))
        
        # Different patterns for different concepts
        if concept_idx % 4 == 0:
            # Horizontal stripes
            pattern[:, ::4, :] = 1
        elif concept_idx % 4 == 1:
            # Vertical stripes
            pattern[:, :, ::4] = 1
        elif concept_idx % 4 == 2:
            # Diagonal pattern
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if (i + j) % 8 == 0:
                        pattern[:, i, j] = 1
        else:
            # Center blob
            center = self.image_size // 2
            y, x = np.ogrid[:self.image_size, :self.image_size]
            mask = (x - center)**2 + (y - center)**2 <= (self.image_size // 4)**2
            pattern[:, mask] = 1
        
        return pattern
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.images[idx]),
                torch.FloatTensor(self.concepts[idx]),
                torch.LongTensor([self.labels[idx]])[0])

# Demonstration and visualization functions
def demonstrate_cbm():
    """Demonstrate Concept Bottleneck Models."""
    print("🧠 Concept Bottleneck Models for Vision Tasks Demo")
    print("=" * 60)
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    train_dataset = SyntheticConceptDataset(num_samples=800, num_concepts=8, num_classes=3)
    val_dataset = SyntheticConceptDataset(num_samples=200, num_concepts=8, num_classes=3)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create and train CBM
    print("\n🏗️  Creating Concept Bottleneck Model...")
    model = ConceptBottleneckModel(
        backbone='resnet18',
        num_concepts=8,
        num_classes=3,
        concept_dim=64
    )
    
    # Set meaningful concept names
    concept_names = [
        "Horizontal Stripes", "Vertical Stripes", "Diagonal Pattern", "Center Blob",
        "Edge Features", "Texture Density", "Color Contrast", "Symmetry"
    ]
    model.set_concept_names(concept_names)
    
    # Train the model
    print("\n🚀 Training CBM...")
    trainer = CBMTrainer(model, concept_weight=1.0, class_weight=1.0)
    trainer.train(train_loader, val_loader, epochs=20, lr=0.001)
    
    # Analyze the trained model
    print("\n📈 Analyzing model interpretability...")
    analyzer = CBMAnalyzer(model)
    
    # Concept importance analysis
    importance_scores = analyzer.analyze_concept_importance(val_loader)
    
    # Visualize results
    visualize_cbm_results(trainer, analyzer, val_loader, importance_scores)
    
    # Demonstrate concept intervention
    print("\n🔧 Demonstrating concept intervention...")
    intervention_model = ConceptInterventionModel(
        backbone='resnet18',
        num_concepts=8,
        num_classes=3
    )
    intervention_model.load_state_dict(model.state_dict())
    
    intervention_analyzer = CBMAnalyzer(intervention_model)
    intervention_results = intervention_analyzer.concept_intervention_analysis(
        val_loader, concept_idx=0, intervention_values=[0.0, 0.5, 1.0])
    
    analyze_interpretability_tradeoffs(trainer, analyzer, val_loader)
    
    return model, trainer, analyzer

def visualize_cbm_results(trainer, analyzer, val_loader, importance_scores):
    """Visualize CBM training results and interpretability."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training curves
    axes[0, 0].plot(trainer.train_losses, label='Training Loss', alpha=0.7)
    axes[0, 0].plot(trainer.val_losses, label='Validation Loss', alpha=0.7)
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(trainer.concept_accuracies, label='Concept Accuracy', alpha=0.7)
    axes[0, 1].plot(trainer.class_accuracies, label='Classification Accuracy', alpha=0.7)
    axes[0, 1].set_title('Accuracy Progress')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Concept importance heatmap
    importance_matrix = np.array([scores for scores in importance_scores.values()])
    sns.heatmap(importance_matrix, 
                xticklabels=[f'C{i}' for i in range(importance_matrix.shape[1])],
                yticklabels=list(importance_scores.keys()),
                annot=True, fmt='.2f', cmap='viridis',
                ax=axes[0, 2])
    axes[0, 2].set_title('Concept Importance by Class')
    
    # Concept space visualization
    concept_2d, true_labels = analyzer.visualize_concept_space(val_loader, max_samples=200)
    scatter = axes[1, 0].scatter(concept_2d[:, 0], concept_2d[:, 1], 
                                c=true_labels, cmap='tab10', alpha=0.7)
    axes[1, 0].set_title('Concept Space Visualization (t-SNE)')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Concept statistics
    concept_stats = analyzer.compute_interpretability_metrics(val_loader)
    axes[1, 1].bar(range(len(concept_stats['mean_activation'])), 
                   concept_stats['mean_activation'])
    axes[1, 1].set_title('Mean Concept Activations')
    axes[1, 1].set_xlabel('Concept Index')
    axes[1, 1].set_ylabel('Mean Activation')
    
    # Concept correlation heatmap
    sns.heatmap(concept_stats['concept_correlation'], 
                annot=False, cmap='RdBu_r', center=0,
                ax=axes[1, 2])
    axes[1, 2].set_title('Concept Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

def analyze_interpretability_tradeoffs(trainer, analyzer, val_loader):
    """Analyze trade-offs between interpretability and accuracy."""
    print("\n🔍 Interpretability vs Accuracy Trade-off Analysis")
    print("-" * 50)
    
    # Get concept statistics
    concept_stats = analyzer.compute_interpretability_metrics(val_loader)
    
    # Compute interpretability metrics
    mean_activations = concept_stats['mean_activation']
    concept_diversity = concept_stats['std_activation']
    concept_correlation = concept_stats['concept_correlation']
    
    # Interpretability scores
    activation_diversity = np.mean(concept_diversity)
    correlation_score = 1 - np.mean(np.abs(concept_correlation - np.eye(len(concept_correlation))))
    confidence_score = np.mean(concept_stats['confidence'])
    
    interpretability_score = (activation_diversity + correlation_score + confidence_score) / 3
    
    # Performance metrics
    final_concept_acc = trainer.concept_accuracies[-1]
    final_class_acc = trainer.class_accuracies[-1]
    
    print(f"📊 Final Performance Metrics:")
    print(f"   Concept Accuracy: {final_concept_acc:.4f}")
    print(f"   Classification Accuracy: {final_class_acc:.4f}")
    print(f"\n🔍 Interpretability Metrics:")
    print(f"   Activation Diversity: {activation_diversity:.4f}")
    print(f"   Concept Independence: {correlation_score:.4f}")
    print(f"   Concept Confidence: {confidence_score:.4f}")
    print(f"   Overall Interpretability: {interpretability_score:.4f}")
    
    # Trade-off analysis
    print(f"\n⚖️  Trade-off Analysis:")
    if interpretability_score > 0.7 and final_class_acc > 0.8:
        print("   ✅ Good balance: High interpretability with strong performance")
    elif interpretability_score > 0.7:
        print("   ⚠️  High interpretability but moderate performance")
    elif final_class_acc > 0.8:
        print("   ⚠️  Strong performance but limited interpretability")
    else:
        print("   ❌ Both interpretability and performance need improvement")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if activation_diversity < 0.3:
        print("   - Consider increasing concept dimensionality")
    if correlation_score < 0.5:
        print("   - Add regularization to reduce concept correlation")
    if confidence_score < 0.5:
        print("   - Increase training epochs or adjust learning rate")
    if final_class_acc < 0.7:
        print("   - Consider adjusting concept-to-class weight ratio")

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run demonstration
    model, trainer, analyzer = demonstrate_cbm()
    
    print("\n✅ Concept Bottleneck Model demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("• Interpretable concept prediction as intermediate step")
    print("• Joint training of concept and classification objectives")
    print("• Concept importance analysis for different classes")
    print("• Concept space visualization using t-SNE")
    print("• Trade-off analysis between interpretability and accuracy")
    print("• Concept intervention capabilities")
    print("• Comprehensive interpretability metrics")
