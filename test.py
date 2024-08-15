import torch
from torch.utils.data import DataLoader, Subset
import os
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr
import numpy as np
import matplotlib.pyplot as plt
from dataloader import CelebADataset

from frequency_network import Luna_Net

#old path
#os.chdir("/users/pgrad/meti/Downloads")
#path = os.getcwd()
#change path for convenience
path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(path, "dataset/medium_images")
mask_path = os.path.join(path, "dataset/medium_masks")

#hyperparameters
batch_size = 16
in_channels = 4
out_channels = 3
factor = 8

#data loading
dataset = CelebADataset(image_dir=dataset_path, mask_dir=mask_path)
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size
#indexing
indices = list(range(dataset_size))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

_, test_dataset = Subset(dataset, train_indices), Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#load the generator into the gpu
gen = Luna_Net(in_channels=in_channels, out_channels=out_channels, factor=factor)
gen.to(device)

#checkpoints
checkpoint_path = "frequency_latest.pth"
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()
else:
    exit("Failed to locate checkpoint")

total_psnr, total_ssim, total_l1 = 0, 0, 0

#metrics computation
def compute_metrics(corrupted_batch, normal_batch):
    psnr, ssim, l1_norm = 0, 0, 0

    corrupted_batch = corrupted_batch.permute(0, 2, 3, 1).cpu().detach().numpy()
    normal_batch = normal_batch.permute(0, 2, 3, 1).cpu().detach().numpy()

    corrupted_y = np.array([rgb2ycbcr(image)[:,:,0] for image in corrupted_batch])
    normal_y = np.array([rgb2ycbcr(image)[:,:,0] for image in normal_batch])

    for corrupted, normal in zip(corrupted_y, normal_y):

        # PSNR
        psnr += peak_signal_noise_ratio(
            normal.astype(np.uint8), corrupted.astype(np.uint8)
        )

        # SSIM
        ssim += structural_similarity(
            normal.astype(np.uint8),
            corrupted.astype(np.uint8),
            #multichannel=True,
            channel_axis=2,
        )

        # L1 Norm
        l1_norm += np.mean(np.abs(normal - corrupted))

    num_images = corrupted_batch.shape[0]
    psnr /= num_images
    ssim /= num_images
    l1_norm /= num_images

    return psnr, ssim, l1_norm


result = 1
inpainted_images = []
ground_truth_images = []

for corrupted_image, normal_image, masks in test_loader:
    _, outputs = gen(corrupted_image.to(device), masks.to(device))
    psnr, ssim, l1_norm = compute_metrics(outputs, normal_image.to(device))

    total_psnr += psnr
    total_ssim += ssim
    total_l1 += l1_norm

    for i, (image, target, mask) in enumerate(
        zip(corrupted_image, normal_image, masks)
    ):
        _, inpainted_img = gen(
            image.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
        )
        inpainted_img = inpainted_img.squeeze(0).cpu().detach()
        result += 1

num_images = len(test_loader)
avg_psnr = total_psnr / num_images
avg_ssim = total_ssim / num_images
avg_l1 = total_l1 / num_images

print("PSNR:", avg_psnr)
print("SSIM:", avg_ssim)
print("L1 Norm:", avg_l1)