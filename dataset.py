import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class PalmBiometricDataset(Dataset):
    """
    Custom PyTorch Dataset for paired Palmprint (RGB) and Palm Vein (NIR) ROI images.
    Expected Directory Structure:
        dataset_root/
            ├── palmprint/
            │   ├── subject_001_1.jpg
            │   └── subject_002_1.jpg
            └── palmvein/
                ├── subject_001_1.jpg
                └── subject_002_1.jpg
    """
    def __init__(self, root_dir, transform_print=None, transform_vein=None):
        self.root_dir = root_dir
        self.print_dir = os.path.join(root_dir, "palmprint")
        self.vein_dir = os.path.join(root_dir, "palmvein")
        
        # Gather image file paths
        self.print_images = sorted(glob.glob(os.path.join(self.print_dir, "*.[jJ][pP][gG]")) + 
                                   glob.glob(os.path.join(self.print_dir, "*.[pP][nN][gG]")))
        self.vein_images = sorted(glob.glob(os.path.join(self.vein_dir, "*.[jJ][pP][gG]")) + 
                                  glob.glob(os.path.join(self.vein_dir, "*.[pP][nN][gG]")))

        # Extract numerical subject/identity labels from file names
        # Example filename: "subject_001_1.jpg" -> ID: 1
        self.labels = [self._extract_label(f) for f in self.print_images]

        # Transforms for data preprocessing
        self.transform_print = transform_print or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_vein = transform_vein or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _extract_label(self, filepath):
        filename = os.path.basename(filepath)
        # Extract integer identity from label pattern (e.g., subject_001_1.jpg -> 1)
        label_str = filename.split('_')[1] if '_' in filename else filename.split('.')[0]
        return int(label_str) - 1  # 0-indexed for PyTorch

    def __len__(self):
        return len(self.print_images)

    def __getitem__(self, idx):
        # Load Palmprint (RGB) and Palm Vein (NIR / Grayscale)
        img_print = Image.open(self.print_images[idx]).convert("RGB")
        img_vein = Image.open(self.vein_images[idx]).convert("L")

        if self.transform_print:
            img_print = self.transform_print(img_print)
        if self.transform_vein:
            img_vein = self.transform_vein(img_vein)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_print, img_vein, label


if __name__ == "__main__":
    print("--- Testing Dataset Pipeline ---")
    # Generate dummy directory structure if running standalone
    os.makedirs("dummy_data/palmprint", exist_ok=True)
    os.makedirs("dummy_data/palmvein", exist_ok=True)

    # Save sample synthetic images for verification
    Image.new('RGB', (128, 128), color='red').save("dummy_data/palmprint/sub_001_1.jpg")
    Image.new('L', (128, 128), color=128).save("dummy_data/palmvein/sub_001_1.jpg")

    dataset = PalmBiometricDataset(root_dir="dummy_data")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for p, v, l in loader:
        print(f"[✓] Palmprint Batch Shape : {p.shape} (Expected: [1, 3, 128, 128])")
        print(f"[✓] Palmvein Batch Shape  : {v.shape} (Expected: [1, 1, 128, 128])")
        print(f"[✓] Class Label Loaded   : {l.item()}")
        break