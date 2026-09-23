import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. MODALITY-SPECIFIC FEATURE EXTRACTION STREAMS
# =====================================================================

class TextureStream(nn.Module):
    """
    Stream A (Palmprint): Processes surface texture and principal line details.
    Uses 3x3 convolutions for fine spatial feature mapping.
    """
    def __init__(self, feature_dim=512):
        super(TextureStream, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.res_block(x)
        x = self.gap(x)
        return torch.flatten(x, 1)


class StructureStream(nn.Module):
    """
    Stream B (Palm Vein): Processes subsurface NIR vascular line topologies.
    Uses larger 5x5 receptive fields to capture continuous vein paths.
    """
    def __init__(self, feature_dim=512):
        super(StructureStream, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.res_block(x)
        x = self.gap(x)
        return torch.flatten(x, 1)


# =====================================================================
# 2. NOVELTY: UNCERTAINTY-AWARE DYNAMIC FUSION MODULE (UADM)
# =====================================================================

class UncertaintyFusionModule(nn.Module):
    """
    Predicts confidence/uncertainty logits for each modality stream.
    Normalizes weights dynamically via Softmax to weight cleaner inputs higher.
    """
    def __init__(self, feature_dim=512):
        super(UncertaintyFusionModule, self).__init__()
        
        # Uncertainty predictor for Palmprint (Stream A)
        self.u_head_p = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # Uncertainty predictor for Palm Vein (Stream B)
        self.u_head_v = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, f_p, f_v):
        # Predict unnormalized confidence scores
        s_p = self.u_head_p(f_p)  # [B, 1]
        s_v = self.u_head_v(f_v)  # [B, 1]

        # Softmax normalization across modalities for dynamic gating
        scores = torch.cat([s_p, s_v], dim=1)  # [B, 2]
        weights = F.softmax(scores, dim=1)    # [B, 2] -> (alpha_p, alpha_v)

        alpha_p = weights[:, 0].unsqueeze(1)  # [B, 1]
        alpha_v = weights[:, 1].unsqueeze(1)  # [B, 1]

        # Dynamically weighted feature fusion
        f_fused = (alpha_p * f_p) + (alpha_v * f_v)
        
        return f_fused, alpha_p, alpha_v


# =====================================================================
# 3. FULL INTEGRATED UAD-NET ARCHITECTURE
# =====================================================================

class UADNet(nn.Module):
    def __init__(self, num_classes=500, feature_dim=512):
        super(UADNet, self).__init__()
        self.texture_stream = TextureStream(feature_dim=feature_dim)
        self.structure_stream = StructureStream(feature_dim=feature_dim)
        self.fusion_module = UncertaintyFusionModule(feature_dim=feature_dim)
        
        # Final Multimodal Classifier Head (Using LayerNorm for batch-size safety)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, img_print, img_vein):
        # 1. Feature Extraction
        f_p = self.texture_stream(img_print)  # [B, 512]
        f_v = self.structure_stream(img_vein) # [B, 512]

        # 2. Dynamic Uncertainty Fusion
        f_fused, alpha_p, alpha_v = self.fusion_module(f_p, f_v)

        # 3. Identity Classification
        logits = self.classifier(f_fused)

        return logits, f_fused, alpha_p, alpha_v


# =====================================================================
# 4. STANDALONE VERIFICATION BLOCK
# =====================================================================

if __name__ == "__main__":
    print("--- Verifying Updated UAD-Net PyTorch Architecture ---")
    
    # Instantiate model
    model = UADNet(num_classes=100, feature_dim=512)
    model.eval()

    # Test batch size of 1 to ensure batch normalization/shape crashes are resolved
    dummy_print = torch.randn(1, 3, 128, 128)
    dummy_vein = torch.randn(1, 1, 128, 128)

    with torch.no_grad():
        logits, f_fused, alpha_p, alpha_v = model(dummy_print, dummy_vein)

    print(f"[✓] Logits Shape        : {logits.shape}  --> Expected: [1, 100]")
    print(f"[✓] Fused Feature Shape : {f_fused.shape} --> Expected: [1, 512]")
    print(f"[✓] Print Weight (alpha_p): {alpha_p.item():.4f}")
    print(f"[✓] Vein Weight  (alpha_v): {alpha_v.item():.4f}")
    print("[✓] Forward pass executed successfully without errors!")