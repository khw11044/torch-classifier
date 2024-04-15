import torch
import os
import csv
import pandas as pd

from dataset import TestDataset
from model import MyModel, TransModel
from activate import demo
from torch.utils.data import DataLoader
from torchvision import transforms
from transform import *

device = torch.device('cuda:0')

IMG_SIZE=64
BATCHSIZE = 128

root = '../open'
train_path = root + '/train.csv'
train_csv = pd.read_csv(train_path)
classes = sorted(list(set(train_csv['label'].unique())))


load_model = '../result/best_model1.pth'

load_submit_csv = '../open/sample_submission.csv'
save_submit_csv = '../result/sample_submission.csv'

num_classes = len(classes)
model = TransModel(num_classes = num_classes).to(device)
model.load_state_dict(torch.load(load_model)['net_dict'])
model = model.to(device)


dataset = TestDataset(root=root,
                    mode='test',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        
                            ])
                        )


test_dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle = False)

preds = demo(model, test_dataloader, device)



test_df = pd.read_csv(load_submit_csv)
print(test_df)
preds = [classes[pred] for pred in preds]
answer = np.array(preds)
test_df.iloc[:,1] = answer
test_df.to_csv(save_submit_csv, index=False)

# df = pd.DataFrame(preds)
# df.to_csv(submit_csv, index=False)
print('complete SAVE csv')

test_df = pd.read_csv(save_submit_csv)

print(test_df)