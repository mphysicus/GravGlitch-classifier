import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from .config import use_scheduler, min_lr, batch_size, num_classes, epochs, learning_rate, train_dir, val_dir, test_dir
from .dataset import gravityspy_loader
from .model import GravitySpyResNet

def train_model(model=None, num_epochs=epochs, optimizer=None):
    if model is None:
        model = GravitySpyResNet()

    if optimizer is None:
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_loss = float('inf')
    checkpoint_dir = '_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize scheduler
    if use_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            eta_min=min_lr
        )
    else:
        scheduler = None

    train_loader = gravityspy_loader(train_dir, shuffle=True, batch_size=batch_size)
    val_loader = gravityspy_loader(val_dir, shuffle=False) if os.path.exists('val') else None
    loss_fn=nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        val_acc = None
        avg_val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_outputs = model(val_images)
                    v_loss = loss_fn(val_outputs, val_labels)
                    
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()
                    val_loss += v_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            model.train()

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # Save best model
        if val_loader and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{checkpoint_dir}/best_weights.pth')

        # Print progress
        log = (f'Epoch [{epoch+1}/{num_epochs}] | '
               f'Train Loss: {avg_train_loss:.4f} | '
               f'Train Acc: {train_acc:.2f}% | '
               f'Val Loss: {avg_val_loss or 0:.4f} | '
               f'Val Acc: {val_acc or 0:.2f}% | '
               f'LR: {current_lr:.6f}')
        print(log)

    # Save final model
    torch.save(model.state_dict(), f'{checkpoint_dir}/final_weights.pth')