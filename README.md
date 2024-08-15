# Frequency Deformable Multi-Head Attention

This project provides two main implementations for image inpainting: **Simplified** and **Frequency**.

## Getting Started

### Training

1. **Download the dataset:**
   [Link to Dataset]

2. **Data Setup:** Place the downloaded dataset folders under the `dataset` directory in the project root, for both implementations.  Your project directory should look like this:

Simplified/
├── dataset/
│ ├── ...
│ └── ...
└── ...

or

Frequency/
├── dataset/
│ ├── ...
│ └── ...
└── ...


3. **Learning Rate Adjustment:** Adjust the learning rate in the training script based on your GPU. If you have a GPU with >40GB VRAM, consider changing the learning rate accordingly.


4. **Start the training run with the following command:**
```python3 main.py```

### Inference

1. **Download Medium Images and Masks:** Download a set of medium-sized images and corresponding masks from the dataset link above.

2. **Place Data in Dataset Folder:** Place these downloaded images and masks under the `dataset` directory.

3. **Run inference with the following command:**

```python3 test.py```