import os
import torch
import torch.utils.data as data
from torchvision.io import read_image


class ImageDataset(data.Dataset):

    def __init__(self, img_dir, transform=None, fraction=1.0, normalization_mode=1):
        """Image dataset for training and validation
        
        Parameters
        ----------
        img_dir : str
            Path to the folder containing the images.
        transform : torchvision.transforms
            Image transformation.
        fraction : float
            Fraction of the dataset to use.
        normalization_mode : int
            1: Normalize the image to [0, 1]
            2: Normalize the image to [-1, 1]
        """
        self.img_dir = img_dir
        self.transform = transform
        self.fraction = fraction
        if normalization_mode not in [1, 2]:
            raise ValueError('Invalid normalization mode')
        else:
            self.normalization_mode = normalization_mode
        self.img_filenames = os.listdir(img_dir)
        self.img_filenames = self.img_filenames[:int(len(self.img_filenames) * self.fraction)]

    def __len__(self):
        return len(self.img_filenames)
        
    def __getitem__(self, idx):
        img_name = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)
        img = img[:3, :, :]
        if self.normalization_mode == 1:
            img = img / 255.0
        elif self.normalization_mode == 2:
            img = (img / 127.5) - 1
        if self.transform:
            img = self.transform(img)
        return img
    
    

def prepare_ImageDataset(img_dir, 
                         batch_size,
                         validation_split,
                         transform=None,
                         seed=123, 
                         fraction=1.0,
                         normalization_mode=1):
    dataset = ImageDataset(img_dir, transform, fraction, normalization_mode)
    val_len = int(len(dataset) * validation_split)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = data.random_split(dataset, 
                                                   lengths=[train_len, val_len], 
                                                   generator=generator)
    train_dataloader = data.DataLoader(train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=True, 
                                       num_workers=4)
    val_dataloader = data.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
    return train_dataset, val_dataset, train_dataloader, val_dataloader
    