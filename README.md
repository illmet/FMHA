# Frequency Deformable Multi-Head Attention

This project provides two main implementations for image inpainting: **Simplified** and **Frequency**.

## Getting Started

### Training

1. **Download the dataset:**
   [https://drive.google.com/drive/folders/1fxrBKYsuFCT6NI2pPgWQN9NloqSdCfMg?usp=drive_link]

2. **Data Setup:** Place the downloaded dataset folders under the `dataset` directory, like so:

Simplified/dataset/

or

Frequency/dataset/


3. **Learning Rate Adjustment:** Adjust the learning rate in the training script based on your GPU. If you have a GPU with >40GB VRAM, consider changing the learning rate accordingly.


4. **Start the training run with the following command:**

```python3 main.py```

### Inference

1. **Download the medium-sized sample:** Download a set of medium-sized images and corresponding masks (in the range of 0.2-0.4) using the same dataset link as above.

2. **Place Data in Dataset Folder:** Place these downloaded images and masks under the `dataset` directory.

3. **Download Checkpoints:**
    [https://drive.google.com/drive/folders/1cTqxXHxRPgYSPpTTFC74pAEhPbBCoEah?usp=drive_link]


4. **Run inference with the following command:**

```python3 test.py```