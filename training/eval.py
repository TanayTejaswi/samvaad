import os
import json
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from training.models import SignNetGRU, SignNetTransformer
from training.dataset import ISLDataset

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    
    # Load dataset
    test_dataset = ISLDataset(args.npz_dir, args.splits_json, split="test", is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Test size: {len(test_dataset)} samples")
    
    # Load vocabulary
    vocab_path = os.path.join(args.run_dir, "vocab.json")
    with open(vocab_path, "r") as f:
        idx_to_label = json.load(f)
        
    num_classes = len(idx_to_label)
    
    # Load model
    model_path = os.path.join(args.run_dir, "best_model.pt")
    if args.model == "gru":
        model = SignNetGRU(input_dim=254, hidden_dim=128, num_classes=num_classes)
    else:
        model = SignNetTransformer(input_dim=254, d_model=128, num_classes=num_classes)
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_top5_acc = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Top-5 Accuracy
            _, top5 = logits.topk(5, 1, True, True)
            top5 = top5.cpu().numpy()
            targets = targets.cpu().numpy()
            for i in range(len(targets)):
                all_top5_acc.append(1 if targets[i] in top5[i] else 0)
                
    top1_acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    top5_acc = np.mean(all_top5_acc) * 100
    
    print("\n--- Results on Signer-Independent Test Set ---")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    
    target_names = [idx_to_label[str(i)] for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("SignNet Confusion Matrix (Test Split)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    cm_path = os.path.join(args.run_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--npz_dir", type=str, default="data/raw_npz")
    parser.add_argument("--splits_json", type=str, default="data/splits/dataset_splits.json")
    parser.add_argument("--model", type=str, choices=["gru", "transformer"], default="gru")
    args = parser.parse_args()
    
    evaluate(args)
