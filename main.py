import torch
from dataset import MyDataset
from model import MyModel
from loss import MyLoss1, MyLoss2
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from activate import train, validation
from torchvision import transforms
from transform import *
from utils import result_save

device = torch.device('cuda:0')


save_path = '../result'
os.makedirs(save_path, exist_ok=True)
save_model = save_path + '/best_model1.pth'

IMG_SIZE=224
BATCHSIZE = 64
LR = 0.01
EPOCHS = 20
train_path = '../Car_Images/train'
test_path = '../Car_Images/test'


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
num_classes = 2 # len(classes)


train_dataset = MyDataset(path=train_path,
                          mode='train',
                          classes=classes,
                          transform=transforms.Compose([
                                Rescale(IMG_SIZE),
                                RandomFlip(0.5),
                                RandomRotate(0.5),
                                RandomErase(0.2),
                                RandomShear(0.2),
                                # RandomCrop(IMG_SIZE),
                                ToTensor(),
                            ])
                        )

validation_dataset = MyDataset(path=train_path,
                               mode='vaildation',
                               classes=classes,
                               transform=transforms.Compose([
                                    Rescale(IMG_SIZE),
                                    RandomFlip(0.5),
                                    RandomRotate(0.5),
                                    RandomErase(0.2),
                                    RandomShear(0.2),
                                    # RandomCrop(IMG_SIZE),
                                    ToTensor(),
                                ])
                            )

model = MyModel(num_classes = num_classes).to(device)
# criterion = MyLoss1(weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32).to(device))
criterion = MyLoss1(weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32)).to(device)
#criterion = MyLoss2(weights = torch.tensor(train_dataset.get_class_weights2(), dtype=torch.float32).to(device))

optimizer = optim.SGD(model.parameters(), lr = LR)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor = 0.5)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20,25,30], gamma=0.1)

train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE, shuffle = True, num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size = BATCHSIZE, shuffle = False)

pre_score = 0

epochs = []
train_loss_list = []
val_loss_list = []
scores_list = []

for epoch in range(EPOCHS):
    print(f'Epoch : {epoch}/{EPOCHS}')
    # 훈련 진행 
    train_loss, train_score = train(model, train_dataloader, criterion, optimizer, device)
    
    val_loss, score = validation(model, validation_dataloader, criterion,  None, device)
    # 5 epoch마다 확인하고 저장 
    if (epoch+1)%5==0:
        if score>pre_score:
            pre_score = score
            model = model.cpu()
            print('---'*10)
            print('best score: {} and save model'.format(score))
            torch.save(model.state_dict(), save_model)
            model = model.to(device)
        scheduler.step(score)

    epochs.append(epoch)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    scores_list.append(score)
    

result_save(epochs,train_loss_list,val_loss_list,scores_list)