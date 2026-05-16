#!/usr/bin/env python3
"""
Complete Model Efficiency Testing Script
Tests reconstruction quality, speed, and performance metrics
"""

import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import glob
import os
import time

print("\n" + "="*70)
print("🧪 MEDICAL IMAGE RECONSTRUCTION - MODEL EFFICIENCY TEST")
print("="*70)

# ==========================================
# 1. LOAD MODEL
# ==========================================
print("\n📦 Loading model...")
model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    activation='sigmoid'
)
model.load_state_dict(torch.load('models/unet_model_weights.pth'))
model.eval()
print("✅ Model loaded successfully!")

# ==========================================
# 2. EXTRACT FEATURES (Same as training)
# ==========================================
def extract_features(img):
    """Extract edge detection features"""
    img_u8 = (img * 255).astype(np.uint8)
    
    canny = cv2.Canny(img_u8, 50, 150) / 255.0
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.clip(cv2.magnitude(sobelx, sobely), 0, 1)
    laplacian = np.clip(np.abs(cv2.Laplacian(img, cv2.CV_64F)), 0, 1)
    
    return np.stack([canny, sobel, laplacian], axis=0)

# ==========================================
# 3. SPEED TEST
# ==========================================
print("\n⚡ Testing inference speed...")
test_input = torch.randn(1, 3, 256, 256)

# Warm up
with torch.no_grad():
    _ = model(test_input)

# Benchmark
start = time.time()
num_iterations = 100
with torch.no_grad():
    for _ in range(num_iterations):
        output = model(test_input)
end = time.time()

avg_time_ms = ((end - start) / num_iterations) * 1000
fps = 1 / (avg_time_ms / 1000)

print(f"  Avg inference time: {avg_time_ms:.2f} ms")
print(f"  FPS (frames/sec): {fps:.1f}")
print(f"  Model size: 94 MB")

# ==========================================
# 4. LOAD TEST IMAGES
# ==========================================
print("\n📁 Loading test images...")
data_folder = 'data'
image_files = sorted(glob.glob(os.path.join(data_folder, '*.*')))

if not image_files:
    print("❌ No images found in data/ folder!")
    exit(1)

# Test on first 5 images
test_images = image_files[:min(5, len(image_files))]
print(f"✅ Found {len(test_images)} test images")

# ==========================================
# 5. BATCH EVALUATION
# ==========================================
print("\n🔍 Evaluating on test images...")
print("-" * 70)

metrics_all = {
    'mse': [],
    'mae': [],
    'rmse': [],
    'ssim': [],
    'psnr': []
}

predictions_list = []
originals_list = []

for idx, img_path in enumerate(test_images, 1):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ⚠️  Skipping {os.path.basename(img_path)} (couldn't read)")
        continue
    
    img = cv2.resize(img, (256, 256)) / 255.0
    
    # Extract features
    input_stack = extract_features(img)
    
    # Predict
    with torch.no_grad():
        output = model(torch.FloatTensor(input_stack).unsqueeze(0))
        prediction = output.squeeze().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(img.flatten(), prediction.flatten())
    mae = mean_absolute_error(img.flatten(), prediction.flatten())
    rmse = np.sqrt(mse)
    ssim_val = ssim(img, prediction, data_range=1.0)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    metrics_all['mse'].append(mse)
    metrics_all['mae'].append(mae)
    metrics_all['rmse'].append(rmse)
    metrics_all['ssim'].append(ssim_val)
    metrics_all['psnr'].append(psnr if psnr != float('inf') else 0)
    
    predictions_list.append(prediction)
    originals_list.append(img)
    
    print(f"\n  Image {idx}: {os.path.basename(img_path)}")
    print(f"    MSE:  {mse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    SSIM: {ssim_val:.4f} (0-1, higher is better)")
    print(f"    PSNR: {psnr:.2f} dB (higher is better)")

# ==========================================
# 6. SUMMARY STATISTICS
# ==========================================
print("\n" + "="*70)
print("📊 OVERALL MODEL EFFICIENCY SUMMARY")
print("="*70)

print(f"\n✅ Tested on {len(test_images)} images\n")

print("Mean Squared Error (MSE):")
print(f"  Average: {np.mean(metrics_all['mse']):.6f}")
print(f"  Min: {np.min(metrics_all['mse']):.6f}")
print(f"  Max: {np.max(metrics_all['mse']):.6f}")

print("\nMean Absolute Error (MAE):")
print(f"  Average: {np.mean(metrics_all['mae']):.6f}")
print(f"  Min: {np.min(metrics_all['mae']):.6f}")
print(f"  Max: {np.max(metrics_all['mae']):.6f}")

print("\nRoot Mean Squared Error (RMSE):")
print(f"  Average: {np.mean(metrics_all['rmse']):.6f}")

print("\nStructural Similarity Index (SSIM) ⭐:")
print(f"  Average: {np.mean(metrics_all['ssim']):.4f}")
print(f"  Min: {np.min(metrics_all['ssim']):.4f}")
print(f"  Max: {np.max(metrics_all['ssim']):.4f}")
print(f"  (Range: 0-1, higher is better)")

print("\nPeak Signal-to-Noise Ratio (PSNR):")
psnr_valid = [p for p in metrics_all['psnr'] if p > 0]
if psnr_valid:
    print(f"  Average: {np.mean(psnr_valid):.2f} dB")
    print(f"  Min: {np.min(psnr_valid):.2f} dB")
    print(f"  Max: {np.max(psnr_valid):.2f} dB")

print("\nInference Speed:")
print(f"  Avg time: {avg_time_ms:.2f} ms per image")
print(f"  FPS: {fps:.1f} images/second")

# ==========================================
# 7. VISUALIZATION
# ==========================================
print("\n📈 Generating visualization...")

num_show = min(3, len(predictions_list))
fig, axes = plt.subplots(num_show, 4, figsize=(16, 4 * num_show))

if num_show == 1:
    axes = np.expand_dims(axes, axis=0)

for i in range(num_show):
    img = originals_list[i]
    pred = predictions_list[i]
    
    # Extract features for display
    features = extract_features(img)
    
    # Column 1: Original
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title(f'Original Image {i+1}', fontsize=10, fontweight='bold')
    axes[i, 0].axis('off')
    
    # Column 2: Canny edges
    axes[i, 1].imshow(features[0], cmap='gray')
    axes[i, 1].set_title(f'Canny Edges', fontsize=10, fontweight='bold')
    axes[i, 1].axis('off')
    
    # Column 3: Prediction
    axes[i, 2].imshow(pred, cmap='gray')
    axes[i, 2].set_title(f'Model Prediction', fontsize=10, fontweight='bold')
    axes[i, 2].axis('off')
    
    # Column 4: Difference
    diff = np.abs(img - pred)
    axes[i, 3].imshow(diff, cmap='hot')
    axes[i, 3].set_title(f'Difference Map', fontsize=10, fontweight='bold')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('results/model_efficiency_test.png', dpi=100, bbox_inches='tight')
print("✅ Visualization saved to results/model_efficiency_test.png")
plt.show()

# ==========================================
# 8. PERFORMANCE RATING
# ==========================================
print("\n" + "="*70)
print("⭐ MODEL PERFORMANCE RATING")
print("="*70)

avg_ssim = np.mean(metrics_all['ssim'])
if avg_ssim >= 0.9:
    rating = "🟢 EXCELLENT - Model performs very well!"
elif avg_ssim >= 0.8:
    rating = "🟡 GOOD - Model performs adequately"
elif avg_ssim >= 0.7:
    rating = "🟠 FAIR - Model shows room for improvement"
else:
    rating = "🔴 POOR - Model needs retraining or architecture change"

print(f"\nSSIM Score: {avg_ssim:.4f}")
print(f"Rating: {rating}")

print(f"\nSpeed: {fps:.1f} FPS ({avg_time_ms:.2f} ms per image)")
if fps >= 10:
    print("⚡ Real-time capable!")
elif fps >= 1:
    print("⏱️  Suitable for batch processing")
else:
    print("🐌 Slow - Consider optimization")

print("\n" + "="*70 + "\n")
