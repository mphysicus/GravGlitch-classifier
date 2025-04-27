import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from config import resize_x, resize_y

original_height = 479
original_width = 569
pad_top = (original_width - original_height) // 2
pad_bottom = (original_width - original_height) - pad_top

gravityspy_transform = transforms.Compose([transforms.Pad((0, pad_top, 0, pad_bottom)),
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2401273101568222, 0.07309725135564804, 0.31978774070739746], std=[0.13638556003570557, 0.13425494730472565, 0.1486215889453888])
    ])

class GravitySpyDataset(ImageFolder):
    def __init__(self, root):
        super().__init__(root, transform=gravityspy_transform)

def gravityspy_loader(dir, shuffle=True, batch_size=None):
    dataset = GravitySpyDataset(dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)