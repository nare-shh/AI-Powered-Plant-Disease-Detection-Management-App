import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preparation import PlantDiseaseDataset, get_transforms
from model import PlantDiseaseModel
from tqdm import tqdm
import mlflow

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Log metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        mlflow.log_metrics({
            "train_loss": running_loss / len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss / len(val_loader),
            "val_acc": val_acc
        }, step=epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {running_loss/len(train_loader):.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%')

def main():
    # MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("plant-disease-detection")
    
    # Data loading
    data_transforms = get_transforms()
    train_dataset = PlantDiseaseDataset('data/train', transform=data_transforms['train'])
    val_dataset = PlantDiseaseDataset('data/val', transform=data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model initialization
    model = PlantDiseaseModel(num_classes=len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    with mlflow.start_run():
        mlflow.log_params({
            "model_type": "ResNet50",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32
        })
        train_model(model, train_loader, val_loader, criterion, optimizer)
        
        # Save model
        torch.save(model.state_dict(), 'saved_models/plant_disease_model.pth')
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()