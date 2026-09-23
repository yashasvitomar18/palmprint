import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom project modules
from model import UADNet
from dataset import PalmBiometricDataset

# =====================================================================
# SUPERVISED CONTRASTIVE LOSS (Adopted from HDC-Net / Paper 2)
# =====================================================================
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Pulls normalized embeddings of the same subject identity closer 
    while pushing features from different identities apart.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize features along embedding dimension
        features = nn.functional.normalize(features, dim=1)
        batch_size = features.shape[0]
        
        # Guard against single-sample batches where contrastive loss is undefined
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Mask out self-contrast (diagonal)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask

        # Compute log-softmax over similarities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean log-likelihood for positive pairs
        denom = mask.sum(1) + 1e-8
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom
        loss = -mean_log_prob_pos.mean()
        return loss


# =====================================================================
# MAIN TRAINING LOOP
# =====================================================================
def train_uadnet(
    data_dir="dummy_data",
    num_classes=100,
    epochs=15,
    batch_size=4,
    lr=1e-3,
    save_path="uadnet_checkpoint.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training UAD-Net on Device: {device}")

    # 1. Prepare Dataset & DataLoader
    dataset = PalmBiometricDataset(root_dir=data_dir)
    
    # drop_last=True prevents batch_size=1 on the final batch
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2 if device.type == "cuda" else 0,
        drop_last=(len(dataset) > batch_size)
    )

    # 2. Instantiate Model & Move to Device
    model = UADNet(num_classes=num_classes, feature_dim=512).to(device)

    # Compile network graph for acceleration on modern GPUs if supported
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[✓] Model compiled successfully using torch.compile()")
        except Exception as e:
            print(f"[!] Compilation skipped: {e}")

    # 3. Loss Functions & Optimizer
    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # PyTorch 2.x updated AMP syntax
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch_idx, (img_print, img_vein, labels) in enumerate(dataloader):
            img_print = img_print.to(device)
            img_vein = img_vein.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Automatic Mixed Precision Forward Pass
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, f_fused, alpha_p, alpha_v = model(img_print, img_vein)
                
                loss_ce = criterion_ce(logits, labels)
                loss_con = criterion_supcon(f_fused, labels)
                
                # Combined Loss: CE + weighted Contrastive Loss
                total_loss = loss_ce + (0.1 * loss_con)

            # Backward Pass & Gradient Scaling
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track Metrics
            running_loss += total_loss.item()
            _, preds = torch.max(logits, 1)
            correct_preds += torch.sum(preds == labels.data).item()
            total_preds += labels.size(0)

        epoch_acc = (correct_preds / total_preds) * 100.0 if total_preds > 0 else 0.0
        avg_loss = running_loss / max(len(dataloader), 1)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # 5. Save Model Checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"[✓] Model checkpoint successfully saved to {save_path}")


if __name__ == "__main__":
    # Auto-generate sample dataset if running standalone for test verification
    if not os.path.exists("dummy_data"):
        print("[*] Generating dummy dataset for execution test...")
        os.system("python dataset.py")

    # Run quick 2-epoch test run
    train_uadnet(data_dir="dummy_data", num_classes=1, epochs=2, batch_size=2)