import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, train_correct, train_total = 0, 0, 0
    preds = []
    targets = []
    for step, (inputs, labels) in enumerate(tqdm(dataloader, desc='Training')):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        # loss = criterion(torch.nn.Softmax(outputs), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


        train_total += labels.size(0)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.detach().cpu().numpy()
        train_correct += ((pred==labels)).sum().item()
        
        for p,t in zip(pred, labels):
            preds.append(p)
            targets.append(t)
            
    f1score = f1_score(np.array(targets), np.array(preds), average='macro') * 100
    train_avg_loss = total_loss / len(dataloader)
    train_avg_accuracy = (train_correct / train_total) * 100

    print("Train Loss : {:.6f}, acc : {:.2f}, f1 : {:.2f}".format(train_avg_loss, train_avg_accuracy, f1score))
    
    return train_avg_loss, train_avg_accuracy, f1score

def validation(model, dataloader, criterion, device):
    model.eval()
    total_loss, vaild_correct, vaild_total = 0, 0, 0
    preds = []
    targets = []
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(tqdm(dataloader, desc='Vaildation')):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            vaild_total += labels.size(0)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.detach().cpu().numpy()
            vaild_correct += ((pred==labels)).sum().item()
            
            for p,t in zip(pred, labels):
                preds.append(p)
                targets.append(t)
                
    f1score = f1_score(np.array(targets), np.array(preds), average='macro') * 100
    vaild_avg_loss = total_loss / len(dataloader)
    vaild_avg_accuracy = (vaild_correct / vaild_total) * 100
    print("Vaild Loss : {:.6f}, acc : {:.2f}, f1 : {:.2f}".format(vaild_avg_loss, vaild_avg_accuracy, f1score))
    return vaild_avg_loss, vaild_avg_accuracy, f1score

def test(model, dataloader, device):
    model.eval()
    test_correct, test_total = 0, 0
    preds = []
    targets = []
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(tqdm(dataloader, desc='Test')):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            test_total += labels.size(0)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.detach().cpu().numpy()
            test_correct += ((pred==labels)).sum().item()
            
            for p,t in zip(pred, labels):
                preds.append(p)
                targets.append(t)
                
        f1score = f1_score(np.array(targets), np.array(preds), average='macro') * 100

        test_avg_accuracy = (test_correct / test_total) * 100
        print("Acc : {:.2f}, f1 : {:.2f}".format(test_avg_accuracy, f1score))
    
    return test_avg_accuracy, f1score

def demo(model, dataloader, device):
    model.eval()

    preds = []

    with torch.no_grad():
        for step, inputs in enumerate(tqdm(dataloader, desc='Demo')):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for p in pred:
                preds.append(p)
                
    return preds
