from torch.utils.data.dataset import Dataset
import torch

class MyDataset(Dataset):

    def __init__(self,img, label):
        self.img = img
        self.label = label


    def __getitem__(self, index):
        image = self.img[index]
        label = self.label[index]

        return image, label

    def __len__(self):
        return len(self.img)