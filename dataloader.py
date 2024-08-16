from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import random
import numpy as np

class BinariseMask(object):
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def __call__(self, mask):
        mask_array = np.array(mask) / 255.0
        binarised = (mask_array > self.threshold).astype(np.uint8) * 255
        return Image.fromarray(binarised)

class CelebADataset(Dataset):
    def __init__(self, image_dir, mask_dir, dilation_range=(9, 49)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dilation_range = dilation_range

        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                BinariseMask(threshold=0.6),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        mask = None

        while mask is None:
            mask_name = random.choice(self.masks)
            mask_path = os.path.join(self.mask_dir, mask_name)

            try:
                mask = Image.open(mask_path).convert("L")
            except:
                continue

        seed = np.random.randint(42)
        random.seed(seed)
        image = self.image_transform(image)
        random.seed(seed)
        mask = self.mask_transform(mask)

        corrupted_image = image * mask

        return corrupted_image, image, mask


if __name__ == "__main__":
    dataset = CelebADataset(
        image_dir="dataset/CelebA-HQ", 
        mask_dir="dataset/irregular_masks"
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    image, _ = next(iter(dataloader))
