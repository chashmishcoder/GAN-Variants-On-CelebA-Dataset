# GAN Variants Comparison on CelebA Dataset

This project implements and compares three GAN variants using the CelebA dataset:
1. Standard GAN with Binary Cross-Entropy Loss
2. Least Squares GAN (LS-GAN)
3. Wasserstein GAN (WGAN)

## Project Structure
```
.
├── config.py           # Configuration parameters
├── data_loader.py      # CelebA dataset loading utilities
├── models/
│   ├── __init__.py
│   ├── discriminator.py
│   ├── generator.py
│   └── gan_variants.py
├── train.py           # Training script
├── evaluate.py        # Evaluation metrics (IS, FID)
└── utils.py           # Utility functions
```

## Setup and Installation

1. Install required packages:
```bash
pip install torch torchvision numpy pandas matplotlib 
pip install scipy scikit-learn pillow
```

2. Download the CelebA dataset:
```bash
python data_loader.py --download
```

3. Train the models:
```bash
python train.py --model bce  # For BCE-GAN
python train.py --model ls   # For LS-GAN
python train.py --model wgan # For WGAN
```

4. Evaluate results:
```bash
python evaluate.py --model all
```

## Dataset
The CelebFaces Attributes Dataset (CelebA) is used for training. It contains 202,599 celebrity face images with variations in pose, background, and clarity.

## Models Implemented
1. **BCE-GAN**: Standard GAN using Binary Cross-Entropy Loss
2. **LS-GAN**: Least Squares GAN for more stable training
3. **WGAN**: Wasserstein GAN for improved convergence

## Evaluation Metrics
- Inception Score (IS)
- Fréchet Inception Distance (FID)
- Visual comparison of generated samples
