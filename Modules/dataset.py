import os
from PIL import Image
from torch.utils.data import Dataset

class TwoStreamCustomDataset(Dataset):
    def __init__(self, data, nir_dir, rgb_dir, transform=None):
        self.data = data
        self.nir_dir = nir_dir
        self.rgb_dir = rgb_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nir_image_path = os.path.join(self.nir_dir, self.data.iloc[idx, 0])
        rgb_image_path = os.path.join(self.rgb_dir, self.data.iloc[idx, 1])
        filename = self.data.iloc[idx, 0]

        nir_image = Image.open(nir_image_path).convert('L')
        rgb_image = Image.open(rgb_image_path).convert('RGB')
        label = self.data.iloc[idx, 3]
        
        # Apply transformations if provided
        if self.transform:
            rgb_image = self.transform(rgb_image)
            nir_image = self.transform(nir_image)

        return filename, nir_image, rgb_image, label

class CustomDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        filename = self.data.iloc[idx, 0]
        # image = Image.open(img_name).convert('L')
        image = Image.open(img_name)
        label = self.data.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return filename, image, label

