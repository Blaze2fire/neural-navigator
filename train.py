"""
Neural Navigator Training Script
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import random
import numpy as np

from data_loader import get_dataloaders
from model import NeuralNavigator, count_parameters

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def apply_augmentation(images):
    """Simple color augmentations."""
    batch_size = images.size(0)
    for i in range(batch_size):
        # Brightness
        if random.random() > 0.5:
            factor = 0.7 + random.random() * 0.6
            images[i] = torch.clamp(images[i] * factor, 0, 1)
        
        # Contrast
        if random.random() > 0.5:
            factor = 0.8 + random.random() * 0.4
            mean = images[i].mean()
            images[i] = torch.clamp((images[i] - mean) * factor + mean, 0, 1)
            
        # Saturation
        if random.random() > 0.5:
            gray = images[i].mean(dim=0, keepdim=True)
            factor = 0.7 + random.random() * 0.6
            images[i] = torch.clamp(images[i] * factor + gray * (1 - factor), 0, 1)

        # Noise
        if random.random() > 0.5:
            noise = torch.randn_like(images[i]) * 0.03
            images[i] = torch.clamp(images[i] + noise, 0, 1)
    return images

def compute_weighted_path_loss(pred_paths, target_paths, time_weight_max=1.0):
    mse = (pred_paths - target_paths) ** 2
    if time_weight_max != 1.0:
        weights = torch.linspace(
            1.0,
            time_weight_max,
            steps=pred_paths.size(1),
            device=pred_paths.device,
        )
        weights = weights / weights.mean()
        mse = mse * weights.view(1, -1, 1)
    return mse.mean()

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    max_grad_norm=1.0,
    augment=True,
    time_weight_max=1.0,
    endpoint_weight=1.0,
):
    model.train()
    total_loss = 0.0
    
    for images, text_tokens, paths in loader:
        if augment:
            images = apply_augmentation(images)
            
        images, text_tokens, paths = images.to(device), text_tokens.to(device), paths.to(device)
        
        optimizer.zero_grad()
        pred_paths, _, aux_target = model(images, text_tokens)
        
        # Loss: weighted path MSE + endpoint MSE
        path_loss = compute_weighted_path_loss(pred_paths, paths, time_weight_max=time_weight_max)
        loss = path_loss + endpoint_weight * criterion(aux_target, paths[:, -1, :])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device, time_weight_max=1.0, endpoint_weight=1.0):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, text_tokens, paths in loader:
            images, text_tokens, paths = images.to(device), text_tokens.to(device), paths.to(device)
            pred_paths, _, aux_target = model(images, text_tokens)
            path_loss = compute_weighted_path_loss(pred_paths, paths, time_weight_max=time_weight_max)
            loss = path_loss + endpoint_weight * criterion(aux_target, paths[:, -1, :])
            total_loss += loss.item()
    return total_loss / len(loader)

def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}, Seed: {args.seed}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, val_split=args.val_split, 
        num_workers=args.num_workers, seed=args.seed
    )
    
    model = NeuralNavigator(num_points=10).to(device)
    print(f"Model Parameters: {count_parameters(model):,}")
    
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    early_stopping = EarlyStopping(patience=args.patience)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                 max_grad_norm=args.max_grad_norm, augment=args.augmentation,
                                 time_weight_max=args.time_weight_max,
                                 endpoint_weight=args.endpoint_weight)
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            time_weight_max=args.time_weight_max,
            endpoint_weight=args.endpoint_weight,
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

        if args.save_every and epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch:03d}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
            
    # Save training curve
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset_repo/data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--time_weight_max', type=float, default=2.0)
    parser.add_argument('--endpoint_weight', type=float, default=1.0)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--save_every', type=int, default=50)
    args = parser.parse_args()
    main(args)
