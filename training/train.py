import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from training.models import SignNetGRU, SignNetTransformer
from training.dataset import ISLDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Setup data
    train_dataset = ISLDataset(args.npz_dir, args.splits_json, split="train", is_training=True)
    val_dataset = ISLDataset(args.npz_dir, args.splits_json, split="val", is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Num classes: {train_dataset.num_classes}")
    
    # Setup model
    if args.model == "gru":
        model = SignNetGRU(input_dim=254, hidden_dim=128, num_classes=train_dataset.num_classes)
    else:
        model = SignNetTransformer(input_dim=254, d_model=128, num_classes=train_dataset.num_classes)
        
    model = model.to(device)
    
    # Loss & Optimizer (Label smoothing for robustness)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Logging dir
    run_dir = f"training/runs/run_{int(time.time())}"
    os.makedirs(run_dir, exist_ok=True)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Save vocab
    with open(os.path.join(run_dir, "vocab.json"), "w") as f:
        json.dump(train_dataset.idx_to_label, f)
        
    print("Starting training...")
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_acc = 100. * train_correct / train_total
        
        # Eval
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            print(f"  -> Saved new best model (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    print(f"Training finished. Best Val Acc: {best_val_acc:.2f}%")
    print(f"Run saved to {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, default="data/raw_npz")
    parser.add_argument("--splits_json", type=str, default="data/splits/dataset_splits.json")
    parser.add_argument("--model", type=str, choices=["gru", "transformer"], default="gru")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    
    train(args)
