import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

def train(model, train_dataloader, criterion, optimizer, device):
    model.train()
    preds = []
    targets = []
    running_loss = 0
    for step, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        # loss = criterion(torch.nn.Softmax(outputs), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        with torch.no_grad():
            pred = torch.softmax(outputs, dim=1).detach().squeeze().cpu().numpy()
            target = labels.detach().cpu().numpy()
            for p,t in zip(pred, target):
                preds.append(np.argmax(p))
                targets.append(t)
    score = f1_score(np.array(targets), np.array(preds), average='macro')
    print("train Loss : {}, score : {}".format(running_loss/len(train_dataloader), score))
    
    return running_loss/len(train_dataloader), score

def validation(model, validation_dataloader, criterion, optimizer, device):
    model.eval()
    preds = []
    targets = []
    running_loss = 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(tqdm(validation_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            pred = torch.softmax(outputs, dim=1).detach().squeeze().cpu().numpy()
            target = labels.detach().cpu().numpy()
            for p,t in zip(pred, target):
                preds.append(np.argmax(p))
                targets.append(t)
    score = f1_score(np.array(targets), np.array(preds), average='macro')
    print("validation Loss : {}, score : {}".format(running_loss/len(validation_dataloader), score))
    return running_loss/len(validation_dataloader), score

def test(model, test_dataloader, criterion, optimizer, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(test_dataloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = torch.softmax(outputs, dim=1).detach().squeeze().cpu().numpy()
            if len(pred.shape)==1:
                preds.append(pred.argmax())
            else:
                for p in pred:
                    preds.append(np.argmax(p))

    return preds
