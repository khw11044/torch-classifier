import torch
from dataset import MyDataset
from model import MyModel, TransModel
from loss import MyLoss1, MyLoss2
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from activate import train, validation, test
from torchvision import transforms
from transform import *
from utils import result_save
import pandas as pd 
from torch.utils.data import Dataset, DataLoader, random_split



device = torch.device('cuda:0')


save_path = '../result'
os.makedirs(save_path, exist_ok=True)
save_model = save_path + '/best_model1.pth'

IMG_SIZE=64
BATCHSIZE = 512
LR = 0.001
start_epoch = 0
EPOCHS = 150
LR_milestones = [100, 125]
LOAD_MODEL=False

root = '../open'
train_path = root + '/train.csv'
train_csv = pd.read_csv(train_path)
test_path = root + '/test.csv'




classes = sorted(list(set(train_csv['label'].unique())))
num_classes = len(classes)



################################### 데이터 로드 ########################

dataset = MyDataset(root=root,
                    mode='train',
                    classes=classes,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(degrees=(-30, 30)),
                        transforms.RandomErasing(p=0.2),
                        transforms.RandomAffine(30, shear=20),
                        # RandomCrop(IMG_SIZE),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        # Normalize 데이터셋에 맞게 조정하기 
                            ])
                        )

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE, shuffle = True, num_workers=0)
validation_dataloader = DataLoader(validation_dataset, batch_size = BATCHSIZE, shuffle = False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=0)



################################### 모델 훈련 설정 ########################



model = TransModel(num_classes = num_classes).to(device)        # MyModel, TransModel

if LOAD_MODEL:
    load_model = save_model
    model.load_state_dict(torch.load(load_model)['net_dict'])
    start_epoch = torch.load(load_model)['epoch']
    
# criterion = MyLoss1(weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32).to(device))
criterion = MyLoss1(weights = torch.tensor(dataset.get_class_weights(), dtype=torch.float32)).to(device)
#criterion = MyLoss2(weights = torch.tensor(train_dataset.get_class_weights2(), dtype=torch.float32).to(device))

optimizer = optim.Adam(model.parameters(), lr = LR) # Adamw

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_milestones, gamma=0.5) # 0.1
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor = 0.5)


train_epochs = []
vaild_epochs = []
train_loss_list = []
val_loss_list = []
train_f1scores_list = []
val_f1scores_list = []
best_score = 0

for epoch in range(start_epoch, EPOCHS):
    print(f'Epoch : {epoch}/{EPOCHS}')
    # 훈련 진행 
    train_loss, train_acc, train_f1 = train(model, train_dataloader, criterion, optimizer, device)
    scheduler.step()
    # 5 epoch마다 확인하고 저장 
    if (epoch)%5==0:
        val_loss, vaild_acc, vaild_f1 = validation(model, validation_dataloader, criterion, device)
        if vaild_f1>best_score:
            best_score = vaild_f1
            model = model.cpu()
            print('---'*10)
            print('best score: {} and save model'.format(vaild_f1))
            torch.save({
                        'epoch': epoch,
                        'net_dict': model.state_dict()
                        }, save_model)
            model = model.to(device)
        
        
        val_loss_list.append(val_loss)
        vaild_epochs.append(int(epoch))
        val_f1scores_list.append(vaild_f1)

    train_loss_list.append(train_loss)
    train_epochs.append(int(epoch))
    train_f1scores_list.append(train_f1)
    
    if (epoch%10==0 or epoch+1==EPOCHS):
        print('Get Graph.....')
        result_save(save_path, epoch, train_epochs,vaild_epochs,train_loss_list,val_loss_list, train_f1scores_list, val_f1scores_list)

test_avg_accuracy, f1score = test(model, test_dataloader, device)

'''
150epoch 
86.85%

'''

