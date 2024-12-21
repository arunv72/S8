import os
print(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from model import CustomCNN
import multiprocessing

multiprocessing.set_start_method('fork')

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)['image']

def get_transforms(dataset_mean, dataset_std):
    train_transform = A.Compose([
        A.RandomCrop(32, 32, padding=4),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            max_holes=3,
            max_height=8,
            max_width=8,
            min_holes=1,
            min_height=4,
            min_width=4,
            fill_value=[int(m * 255) for m in dataset_mean],
            p=0.2
        ),
        A.Normalize(mean=dataset_mean, std=dataset_std),
        ToTensorV2()
    ])
    
    test_transform = A.Compose([
        A.Normalize(mean=dataset_mean, std=dataset_std),
        ToTensorV2()
    ])
    
    return AlbumentationsTransform(train_transform), AlbumentationsTransform(test_transform)

def train_model(model, train_loader, test_loader, device, epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simpler step-based learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.5
    )
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Best Accuracy: {best_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
        
        if best_acc >= 85.0:
            print(f"Reached target accuracy of 85% at epoch {epoch+1}")
            break
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CIFAR-10 dataset mean and std
    dataset_mean = (0.4914, 0.4822, 0.4465)
    dataset_std = (0.2470, 0.2435, 0.2616)
    
    train_transform, test_transform = get_transforms(dataset_mean, dataset_std)
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                   transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                  transform=test_transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, 
                                             shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, 
                                            shuffle=False, num_workers=0)
    
    model = CustomCNN().to(device)
    
    # Print model summary and parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    if total_params > 200000:
        raise ValueError("Model has more than 200k parameters!")
    
    trained_model = train_model(model, train_loader, test_loader, device)
    
    # Save the model
    torch.save(trained_model.state_dict(), 'custom_cnn.pth')

if __name__ == '__main__':
    main()