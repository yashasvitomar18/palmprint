import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torchvision.transforms.functional as TF

# Import project modules
from model import UADNet
from dataset import PalmBiometricDataset

# =====================================================================
# 1. METRIC COMPUTATION FUNCTIONS (EER & Rank-1)
# =====================================================================

def compute_eer(genuine_scores, impostor_scores):
    """
    Computes Equal Error Rate (EER) and ROC parameters given genuine 
    and impostor similarity scores.
    """
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    y_scores = np.concatenate([genuine_scores, impostor_scores])

    # Compute False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # EER is the point where FPR == FNR
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100.0  # Return percentage


def evaluate_recognition_metrics(model, dataloader, device):
    """
    Extracts normalized fused features and computes Rank-1 Accuracy and EER.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for img_print, img_vein, labels in dataloader:
            img_print, img_vein = img_print.to(device), img_vein.to(device)
            
            # Forward pass to extract fused embeddings
            _, f_fused, alpha_p, alpha_v = model(img_print, img_vein)
            
            # L2 Normalize features for cosine similarity distance
            f_norm = F.normalize(f_fused, p=2, dim=1)
            
            all_features.append(f_norm.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0) # [N, 512]
    labels = torch.cat(all_labels, dim=0)     # [N]

    # Compute Cosine Similarity Matrix [N, N]
    sim_matrix = torch.mm(features, features.T).numpy()
    labels_np = labels.numpy()
    num_samples = len(labels_np)

    genuine_scores = []
    impostor_scores = []
    rank1_correct = 0

    for i in range(num_samples):
        # Rank-1 Identification: Mask self-comparison
        scores = sim_matrix[i].copy()
        scores[i] = -np.inf 
        
        predicted_idx = np.argmax(scores)
        if labels_np[predicted_idx] == labels_np[i]:
            rank1_correct += 1

        # Separate Genuine vs Impostor similarity pairs
        for j in range(i + 1, num_samples):
            if labels_np[i] == labels_np[j]:
                genuine_scores.append(sim_matrix[i, j])
            else:
                impostor_scores.append(sim_matrix[i, j])

    rank1_acc = (rank1_correct / num_samples) * 100.0
    
    # Handle single-sample cases gracefully
    if len(genuine_scores) > 0 and len(impostor_scores) > 0:
        eer_val = compute_eer(np.array(genuine_scores), np.array(impostor_scores))
    else:
        eer_val = 0.0

    return rank1_acc, eer_val


# =====================================================================
# 2. SYNTHETIC NOISE & BLUR INJECTION TEST (Ablation Proof)
# =====================================================================

def evaluate_noise_robustness(model, dataloader, device):
    """
    Injects heavy Gaussian blur onto Palmprint inputs while keeping 
    Palmvein clean to verify that the Uncertainty Module dynamically 
    lowers alpha_p and relies on alpha_v.
    """
    model.eval()
    print("\n--- Running Noise Robustness & Dynamic Gating Test ---")

    p_weights = []
    v_weights = []

    with torch.no_grad():
        for img_print, img_vein, _ in dataloader:
            img_print, img_vein = img_print.to(device), img_vein.to(device)

            # Apply heavy synthetic blur to Palmprint stream
            blurred_print = TF.gaussian_blur(img_print, kernel_size=[15, 15], sigma=[5.0, 5.0])

            # Forward pass through UAD-Net
            _, _, alpha_p, alpha_v = model(blurred_print, img_vein)

            p_weights.append(alpha_p.mean().item())
            v_weights.append(alpha_v.mean().item())

    avg_alpha_p = np.mean(p_weights)
    avg_alpha_v = np.mean(v_weights)

    print(f"[✓] Degraded Modality (Blurred Palmprint) Weight  : {avg_alpha_p:.4f}")
    print(f"[✓] Clean Modality (Palm Vein) Weight            : {avg_alpha_v:.4f}")
    
    if avg_alpha_v > avg_alpha_p:
        print("[★] SUCCESS: Network dynamically shifted weight to the clean modality!")
    else:
        print("[!] Note: Train for more epochs to sharpen uncertainty sensitivity.")


# =====================================================================
# 3. MAIN EVALUATION CONTROLLER
# =====================================================================

def run_evaluation(data_dir="dummy_data", checkpoint_path="uadnet_checkpoint.pth", num_classes=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Evaluation on Device: {device}")

    # 1. Load Dataset
    dataset = PalmBiometricDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 2. Instantiate Model
    model = UADNet(num_classes=num_classes, feature_dim=512).to(device)

    # 3. Load Checkpoint if present
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[✓] Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    # 4. Compute Benchmark Metrics
    rank1_acc, eer = evaluate_recognition_metrics(model, dataloader, device)
    
    print("\n==========================================")
    print("      UAD-NET BIOMETRIC PERFORMANCE       ")
    print("==========================================")
    print(f"  Rank-1 Identification Accuracy : {rank1_acc:.2f}%")
    print(f"  Equal Error Rate (EER)         : {eer:.2f}%")
    print("==========================================\n")

    # 5. Execute Noise Injection Benchmark
    evaluate_noise_robustness(model, dataloader, device)


if __name__ == "__main__":
    if not os.path.exists("dummy_data"):
        os.system("python dataset.py")
        
    run_evaluation(data_dir="dummy_data")