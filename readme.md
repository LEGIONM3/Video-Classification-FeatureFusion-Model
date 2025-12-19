# Feature Fusion Network

## Model Architecture
- **Type**: Multi-Modal Hybrid (CNN + Transformer)
- **Pathway 1 (Spatial)**: ResNet3D (r3d_18) for robust localized feature extraction.
- **Pathway 2 (Spatiotemporal)**: TimeSformer (Transformer) block dealing with patches and frames to capture long-range dependencies.
- **Fusion**: Late fusion via concatenation of flattened feature vectors (512 features from CNN + 256 features from Transformer).
- **Classification Head**: MLP mapping fused features to binary classes.

## Dataset Structure
Expects `Dataset` folder in parent directory.
```
Dataset/
├── violence/
└── no-violence/
```

## How to Run
1. Install dependencies: `torch`, `opencv-python`, `scikit-learn`, `numpy`, `torchvision`.
2. Run `python train.py`.
