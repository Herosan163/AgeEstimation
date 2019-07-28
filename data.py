import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):

    def __init__(self, filepath_list, transform=None):
        
        self.images = []
        self.labels = []
        for filepath in filepath_list:
            basename = os.path.basename(filepath)
            self.labels.append(int(basename[4:6]))
            img = np.array(Image.open(filepath).convert('RGB'))
            self.images.append(img)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.transform = transform

    def __len__(self):

        return self.images.shape[0]

    def __getitem__(self, index):

        img = self.images[index]
        # img = self.images[index].astype(np.float32)
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        sample = {'image': img, 'label': label}
        return sample       
