# Medical Image Reconstruction - U-Net Model

A deep learning project for medical image reconstruction using U-Net architecture with edge detection features (Canny, Sobel, Laplacian).

## 📋 Project Overview

This project implements a U-Net based model that reconstructs high-quality medical images from edge-detected features. The model processes 3-channel input (combining Canny, Sobel, and Laplacian edge detection) to produce reconstructed medical images.

**Model Architecture:**
- Encoder: ResNet34 backbone
- Decoder: U-Net architecture with skip connections
- Input: 3 channels (edge detection features)
- Output: 1 channel (reconstructed image)
- Activation: Sigmoid

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Mostov98/medical-image-reconstruction.git
cd medical-image-reconstruction
```

### 2. Install Dependencies
```bash
pip install torch torchvision opencv-python numpy matplotlib segmentation-models-pytorch
```

### 3. Prepare Your Data
Place your medical images in the `data/` folder:
```
data/
├── image_1.jpg
├── image_2.jpg
├── ...
└── image_10.jpg
```

### 4. Train the Model
```bash
python project_model.py
```

This will:
- Load and preprocess images from the `data/` folder
- Train for 100 epochs
- Save the trained model to `models/unet_model_weights.pth`
- Generate and save results visualization to `results/reconstruction_results.png`

## 📦 Using the Pre-trained Model

If the trained model is already available, you can load and use it for inference:

```python
import torch
import segmentation_models_pytorch as smp

# Load the pre-trained model
model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    activation='sigmoid'
)

# Load weights
model.load_state_dict(torch.load('models/unet_model_weights.pth'))
model.eval()

# Use for inference
with torch.no_grad():
    # your_input_image should be shape (1, 3, 256, 256)
    output = model(your_input_image)
```

## ⚙️ Configuration

Edit settings in `project_model.py`:

```python
START_IMG = 0         # First image index to load
END_IMG = 10          # Last image index to load (exclusive)
EPOCHS = 100          # Number of training epochs
BATCH_SIZE = 2        # Batch size for training
LEARNING_RATE = 0.001 # Learning rate for Adam optimizer
```

## 📊 Project Structure

```
medical-image-reconstruction/
├── data/                              # Training images (not in repo)
├── models/
│   └── unet_model_weights.pth        # Trained model weights
├── results/
│   └── reconstruction_results.png    # Visualization of results
├── project_model.py                  # Main training script
├── run_complete_pipeline.py          # Automated training pipeline
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## 🔄 Feature Extraction

The model uses three edge detection techniques as input features:

1. **Canny Edge Detection**: Detects sharp edges and contours
2. **Sobel Edge Detection**: Computes edge gradients (X and Y)
3. **Laplacian Edge Detection**: Second derivative edge detection

These 3 channels are stacked and fed into the U-Net model.

## 📈 Model Performance

- **Loss Function**: L1 Loss
- **Optimizer**: Adam (lr=0.001)
- **Input Size**: 256×256 pixels
- **Batch Size**: 2
- **Epochs**: 100

## 🎯 Next Steps

### Option 1: Share on Hugging Face Hub (Recommended)
```bash
pip install huggingface_hub

python
>>> from huggingface_hub import model_info
>>> # Upload your model to Hugging Face for easy sharing
```

### Option 2: Deploy as Web API
Use frameworks like FastAPI to create a REST API endpoint for inference.

### Option 3: Package as Python Library
Create a pip-installable package for easier distribution.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Mostov98** - Medical Image Reconstruction Project

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

---

**Last Updated**: 2026-05-15

For more information, visit the [GitHub Repository](https://github.com/Mostov98/medical-image-reconstruction)
