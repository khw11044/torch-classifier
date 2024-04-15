import os
import glob
import csv
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from transform import *
import pandas as pd

class TestDataset(Dataset):
    def __init__(self, path='./dataset/test', mode='test', transform=None):
        super(TestDataset, self).__init__()

        self.images = []
        if mode=='test':
            self.images = sorted(glob.glob(path+'/**', recursive=True))
        else:
            raise Exception('Are u sure test?')
        
        while len(self.images[0].split('/')[-1])<4 or self.images[0].split('/')[-1][-3:]!='jpg':
            self.images.pop(0)
        
        self.transform = transform        
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = self.images[idx]
        img = self.read_image(img)
        
        if self.transform:
            img = self.transform(img)

        return img      # torch.Size([3, 224, 224])
    
    def read_image(self, img):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
        


class MyDataset(Dataset):
    def __init__(self, root='../open', mode='train', classes=None, transform=None):
        super(MyDataset, self).__init__()
        if classes is None:
            raise Exception('needed classes')


        data_csv = pd.read_csv(root + '/train.csv')
        # classes = sorted(list(set(data_csv['label'].unique())))
        TOTAL_LEN = len(data_csv)
        self.mode = mode
        self.labels = []
        self.images = []
        self.index_labels = {cls:0 for cls in classes}
        for i, cls in enumerate(classes):
            self.index_labels[cls]=i
        
        
        for i in range(TOTAL_LEN):
            image_path1 = '.'.join(data_csv.loc[i]['img_path'].split('.')[1:])
            image_path2 = '.'.join(data_csv.loc[i]['upscale_img_path'].split('.')[1:])
            label = data_csv.loc[i]['label']
            
            image_path1 = root + image_path1
            image_path2 = root + image_path2
            
            self.images.append(image_path1)
            self.labels.append(label)
            
            self.images.append(image_path2)
            self.labels.append(label)
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img, label = self.images[idx], self.labels[idx]
        img = self.read_image(img)
        label = self.convert_label(label)

        # sample = {'image': img, 'label': label}

        if self.transform:
            img = self.transform(img)

        return img, label   # torch.Size([3, 224, 224])

    def read_image(self, img):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def convert_label(self, label):
        return self.index_labels[label]

    def get_class_weights(self):
        weights = [0 for _ in range(len(self.index_labels))]
        for label in self.labels:
            weights[self.index_labels[label]]+=1
        weights = np.array(weights)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        return weights


class TestDataset(Dataset):
    def __init__(self, root='../open', mode='test', transform=None):
        super(TestDataset, self).__init__()

        data_csv = pd.read_csv(root + '/test.csv')

        TOTAL_LEN = len(data_csv)
        self.mode = mode
        self.images = []

        
        for i in range(TOTAL_LEN):
            image_path1 = '.'.join(data_csv.loc[i]['img_path'].split('.')[1:])
            image_path1 = root + image_path1
            self.images.append(image_path1)
            
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img= self.images[idx]
        img = self.read_image(img)

        # sample = {'image': img, 'label': label}

        if self.transform:
            img = self.transform(img)

        return img   # torch.Size([3, 224, 224])

    def read_image(self, img):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_class_weights(self):
        weights = [0 for _ in range(len(self.index_labels))]
        for label in self.labels:
            weights[self.index_labels[label]]+=1
        weights = np.array(weights)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        return weights


def calculate_norm(dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()
    
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)


def mean_std(train_dataloader):
    mean0 = 0
    mean1 = 0
    mean2 = 0
    std0 = 0
    std1 = 0
    std2 = 0

    for image, _ in train_dataloader:
        mean0+=image[:,0,:,:].mean()
        mean1+=image[:,1,:,:].mean()
        mean2+=image[:,2,:,:].mean()
        std0+=image[:,0,:,:].std()
        std1+=image[:,1,:,:].std()
        std2+=image[:,2,:,:].std()

    print(mean0/len(train_dataloader))
    print(mean1/len(train_dataloader))
    print(mean2/len(train_dataloader))
    print(std0/len(train_dataloader))
    print(std1/len(train_dataloader))
    print(std2/len(train_dataloader))
    
    
if __name__=='__main__':

    classes = set()

    train_path = '../dataset/train'
    test_path = '../dataset/test'

    total_train_num = 0
    total_test_num = 0
    for label in os.listdir(train_path):
        classes.add(label)
        image_num = len(os.listdir(os.path.join(train_path,label)))
        total_train_num += image_num
        print('train dataset size : {} -> {}'.format(label,image_num))

    train_dataset = MyDataset(path=train_path,
                                transform=transforms.Compose([
                                Rescale(224),
                                RandomFlip(0.5),
                                RandomRotate(0.5),
                                RandomErase(0.2),
                                RandomShear(0.2),
                                # RandomCrop(IMG_SIZE),
                                ToTensor(),
                                ])
                            )
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    mean_std(train_dataloader)
