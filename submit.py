import torch
import os
import csv

from dataset import TestDataset
from model import MyModel
from activate import test
from torch.utils.data import DataLoader
from torchvision import transforms
from transform import *

device = torch.device('cuda:0')

save_model = '../result/best_model1.pth'
train_path = '../dataset/train'
test_path = '../dataset/test'

submit_csv = '../result/result.csv'
IMG_SIZE=224
BATCHSIZE = 2

def get_classes(train_path, test_path):
    classes = set()

    total_train_num = 0
    total_test_num = 0
    for label in os.listdir(train_path):
        classes.add(label)
        image_num = len(os.listdir(os.path.join(train_path,label)))
        total_train_num += image_num
        print('train dataset size : {} -> {}'.format(label,image_num))
    for label in os.listdir(test_path):
        image_num = len(os.listdir(os.path.join(test_path,label)))
        total_test_num += image_num
        print('test dataset size : {} -> {}'.format(label,image_num))
    print()
    print('total train dataset : {} \t total vail dataset : {} \t total test dataset : {}'.format(total_train_num*0.9, total_train_num*0.1, total_test_num))      
    return classes

classes = get_classes(train_path, test_path)
num_classes = 7

test_dataset = TestDataset(path=test_path,
                          mode='test',
                          transform=transforms.Compose([
                                Rescale(IMG_SIZE),
                                ToTensor(),
                            ])
                        )

model = MyModel(num_classes = num_classes)
model.load_state_dict(torch.load(save_model))
model = model.to(device)
test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle = False)

preds = test(model, test_dataloader, None,  None, device)


import pandas as pd

test_df = pd.read_csv("../dataset/test_answer_sample_.csv")

answer = np.array(preds)
test_df.iloc[:,1] = answer
test_df.to_csv(submit_csv, index=False)

# df = pd.DataFrame(preds)
# df.to_csv(submit_csv, index=False)
print('complete SAVE csv')