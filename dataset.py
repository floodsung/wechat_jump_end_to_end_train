import torch
import json
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

data_folder = "./dataset/images"
press_times = json.load(open("./dataset/dataset.json"))

image_roots = [os.path.join(data_folder,image_file) \
                for image_file in os.listdir(data_folder)]


class JumpDataset(Dataset):

    def __init__(self,transform = None):
        self.image_roots = image_roots
        self.press_times = press_times
        self.transform = transform

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self,idx):
        image_root = self.image_roots[idx]
        image_name = image_root.split("/")[-1]
        image = Image.open(image_root)
        image = image.convert('RGB')
        image = image.resize((224,224), resample=Image.LANCZOS)
        #image = np.array(image, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        press_time = self.press_times[image_name]

        return image,press_time

def jump_data_loader():
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    dataset =  JumpDataset(transform=transform)
    return DataLoader(dataset,batch_size = 32,shuffle = True)


