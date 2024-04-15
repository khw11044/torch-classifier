import os
import csv
import torch
import cv2
import matplotlib.pyplot as plt

def result_save(save_path, epoch, train_epochs,vaild_epochs, train_loss_list,val_loss_list,train_f1scores_list, val_scores_list):
    plt.figure(figsize=(12,6))

    plt.subplot(1, 2, 1)
    plt.plot(train_epochs,train_loss_list,label='training_loss')
    plt.plot(vaild_epochs,val_loss_list, label='vaildation_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_epochs,train_f1scores_list,label='train_F1')
    plt.plot(vaild_epochs,val_scores_list,label='vaild_F1')
    plt.xlabel('epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.savefig(save_path + '/F1_score_{}.png'.format(int(epoch)))

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

def train_validation_split(train_csv_path, validation_csv_path, split_ratio, train_path = '../train'):
    train_csv = csv.writer(open(train_csv_path, 'w', encoding='utf-8-sig', newline=''))
    validation_csv = csv.writer(open(validation_csv_path, 'w', encoding='utf-8-sig', newline=''))
    index = 1
    #X_train = []
    #Y_train = []
    for label in os.listdir(train_path):
        image_root_path = os.path.join(train_path, label)
        for image_name in os.listdir(image_root_path):
            image_path = os.path.join(image_root_path, image_name)
            if index%(int(1/split_ratio))!=0:
                train_csv.writerow([image_path, label])
            else:
                validation_csv.writerow([image_path, label])
            index+=1

def test_csv(test_csv_path, test_path='../test'):
    test_csv = csv.writer(open(test_csv_path, 'w', encoding='utf-8-sig', newline=''))
    for label in os.listdir(test_path):
        image_root_path = os.path.join(test_path, label)
        for image_name in os.listdir(image_root_path):
            image_path = os.path.join(image_root_path, image_name)
            test_csv.writerow([image_path, label])

if __name__=='__main__':
    '''
    train_csv_path = '../train.csv'
    validation_csv_path = '../validation.csv'
    test_csv_path = '../test.csv'
    split_ratio = 0.1
    
    train_validation_split(train_csv_path, validation_csv_path, split_ratio)
    
    test_csv(test_csv_path)
    '''


