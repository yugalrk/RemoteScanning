from torchvision import transforms
import torch

def replicate_channels(image):
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image

def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(replicate_channels),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def collate_fn(batch):
    return tuple(zip(*batch))
