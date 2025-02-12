import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class PairedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, label):
        seed = torch.random.seed()
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.manual_seed(seed)
        label = self.transform(label)
        return image, label


class PairedTripleTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, label, mask):
        seed = torch.random.seed()
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.manual_seed(seed)
        label = self.transform(label)
        torch.manual_seed(seed)
        mask = self.transform(mask)
        return image, label, mask


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = PairedTransform(transform=transform)
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        self.images.sort()
        self.labels.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')

        image, label = self.transform(image, label)

        return image, label


class SegmentationTripleDataset(SegmentationDataset):

    def __init__(self, image_dir, label_dir, mask_dir, transform=None):
        super().__init__(image_dir, label_dir, transform)
        self.mask_dir = mask_dir
        self.masks = os.listdir(mask_dir)
        self.transform = PairedTripleTransform(transform)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image, label, mask = self.transform(image, label, mask)
        mask = (mask*2).type(torch.LongTensor)
        label = label.type(torch.LongTensor)

        merged_mask = torch.where(label == 1, 1, mask)

        return image, merged_mask.type(torch.LongTensor)