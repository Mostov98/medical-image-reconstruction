import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import glob
import os

# ==========================================
# SETTINGS
# ==========================================
START_IMG = 0      # Start from image 0
END_IMG = 10       # Use all 10 images
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 0.001

# ==========================================
# 1. DATASET CLASS: Extract Features
# ==========================================
class MedicalMultiFeatureDataset(Dataset):
    """Load images and extract edge features"""
    def __init__(self, images_np):
        self.images = images_np

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        orig = self.images[idx]
        orig_u8 = (orig * 255).astype(np.uint8)

        # Extract 3 edge detection features
        # 1. Canny edges
        canny = cv2.Canny(orig_u8, 50, 150) / 255.0
        
        # 2. Sobel edges
        sobelx = cv2.Sobel(orig, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(orig, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.clip(cv2.magnitude(sobelx, sobely), 0, 1)
        
        # 3. Laplacian edges
        laplacian = np.clip(np.abs(cv2.Laplacian(orig, cv2.CV_64F)), 0, 1)

        # Stack into 3-channel input
        input_stack = np.stack([canny, sobel, laplacian], axis=0)
        return torch.FloatTensor(input_stack), torch.FloatTensor(orig).unsqueeze(0)


# ==========================================
# 2. IMAGE LOADER
# ==========================================
def load_specific_images(folder_path, start, end):
    """Load images from folder and resize to 256x256"""
    all_images = []
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
    selected_files = image_files[start:end]

    if not selected_files:
        return None

    for f in selected_files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Normalize to 0-1 range
            all_images.append(cv2.resize(img, (256, 256)) / 255.0)
    
    return np.array(all_images)


# ==========================================
# 3. TRAIN THE MODEL
# ==========================================
folder_path = "data"
images_np = load_specific_images(folder_path, START_IMG, END_IMG)

if images_np is not None:
    print(f"✅ Loaded {len(images_np)} images")
    
    # Create dataset and dataloader
    dataset = MedicalMultiFeatureDataset(images_np)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create U-Net model
    model = smp.Unet(
        encoder_name="resnet34",
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    # Train
    print(f"Training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        for inputs, originals in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, originals)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}")

    print("✅ Training complete!")
    
    # ==========================================
    # 4. SAVE MODEL WEIGHTS
    # ==========================================
    os.makedirs('models', exist_ok=True)
    model_path = 'models/unet_model_weights.pth'
    torch.save(model.state_dict(), model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Model saved to {model_path} ({model_size:.2f} MB)")

    # ==========================================
    # 5. VISUALIZE RESULTS
    # ==========================================
    model.eval()
    num_show = len(dataset)
    fig, axs = plt.subplots(num_show, 5, figsize=(18, 3.5 * num_show))

    if num_show == 1:
        axs = np.expand_dims(axs, axis=0)

    with torch.no_grad():
        for i in range(num_show):
            test_in, test_orig = dataset[i]
            prediction = model(test_in.unsqueeze(0))[0][0]

            # Column 1: Canny edges
            axs[i, 0].imshow(test_in[0], cmap='gray')
            axs[i, 0].axis('off')
            if i == 0:
                axs[i, 0].set_title("1. Canny Edges")

            # Column 2: Sobel edges
            axs[i, 1].imshow(test_in[1], cmap='gray')
            axs[i, 1].axis('off')
            if i == 0:
                axs[i, 1].set_title("2. Sobel Edges")

            # Column 3: Laplacian edges
            axs[i, 2].imshow(test_in[2], cmap='gray')
            axs[i, 2].axis('off')
            if i == 0:
                axs[i, 2].set_title("3. Laplacian Edges")

            # Column 4: Model prediction
            axs[i, 3].imshow(prediction.cpu().numpy(), cmap='gray')
            axs[i, 3].axis('off')
            if i == 0:
                axs[i, 3].set_title("Model Output")

            # Column 5: Original image
            axs[i, 4].imshow(test_orig[0], cmap='gray')
            axs[i, 4].axis('off')
            if i == 0:
                axs[i, 4].set_title("Original (Ground Truth)")

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    results_path = 'results/reconstruction_results.png'
    plt.savefig(results_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"✅ Results saved to {results_path}")

else:
    print("❌ No images found! Check folder path and START/END indices")
