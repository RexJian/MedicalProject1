import copy
import math
import os
import time

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from time import sleep

from Data_Processing import CovidCT

# tensorboard --logdir=runs

CUDA_DEVICES = 0
init_lr = 0.001

# Save model every 5 epochs
checkpoint_interval = 10
if not os.path.isdir('./Checkpoint/'):
    os.mkdir('./Checkpoint/')


# Setting learning rate operation
def adjust_lr(optimizer, epoch):
    # 1/10 learning rate every 5 epochs
    lr = init_lr * (0.1 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def data_augmentation(dir_path):
    img_dir_path= os.getcwd() + dir_path
    clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(3,3))
    cannot_read=0
    save_pth_list=[]
    clahe_img_list=[]
    for img_name in tqdm(os.listdir(dir_path)):
        img_path=img_dir_path+'/'+img_name
        img=cv2.imread(img_path)
        if(img is None):
            cannot_read+=1
        else:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            clahe_img_list.append(clahe.apply(img))
            new_img_name=img_name.replace('.png','_1.png')
            save_pth_list.append(img_dir_path+'/'+new_img_name)
    sleep(0.01)

            # cv2.imwrite(save_pth,clahe_img)
    for clahe_img,save_pth in zip(clahe_img_list,save_pth_list):
        cv2.imwrite(save_pth,clahe_img)
    print("CannotReadImg:"+str(cannot_read))
    print('Augmentation Complete')


    print(img)
def train():
    # If out of memory , adjusting the batch size smaller
    data_transform = {
        'train':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }
    trainset = CovidCT(r"C:/Users/dsp523/Desktop/MIP_Project1/", "train", data_transform['train'])
    # trainset = CovidCT(f"{os.getcwd()+'./Dataset/'}", "train", data_transform['train'])
    train_dl = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    validset = CovidCT(r"C:/Users/dsp523/Desktop/MIP_Project1/", "valid", data_transform['validation'])
    # validset = CovidCT(f"{os.getcwd()+'./Dataset/'}", "valid", data_transform['validation'])
    valid_dl = DataLoader(validset, batch_size=32, shuffle=False, num_workers=4)
    classes = ['1NonCOVID','2COVID']

    model=models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(CUDA_DEVICES)

    print(model)
    print("==========")

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    model = model.cuda(CUDA_DEVICES)

    writer = SummaryWriter()

    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training epochs
    num_epochs = 40
    criterion = nn.CrossEntropyLoss()

    # Optimizer setting
    optimizer = torch.optim.Adam(params=model.fc.parameters())

    # Log
    with open('TrainingAccuracy.txt','w') as fAcc:
        print('Accuracy\n', file = fAcc)
    with open('TrainingLoss.txt','w') as fLoss:
        print('Loss\n', file = fLoss)


    warmup_steps= 1000
    total_steps= num_epochs*len(train_dl)/32
    lambda1=lambda cur_iter: cur_iter / warmup_steps if cur_iter<warmup_steps \
        else (0.5*(1+math.cos(math.pi*0.5*2*((cur_iter-warmup_steps)/(total_steps-warmup_steps)))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda1)


    for epoch in range(num_epochs):
        model.train()
        localtime = time.asctime( time.localtime(time.time()) )
        print('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime))
        print('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime)))

        training_loss = 0.0
        training_corrects = 0
        #adjust_lr(optimizer, epoch)

        for i, (inputs, labels) in (enumerate(tqdm(train_dl))):

            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            training_loss += float(loss.item() * inputs.size(0))
            training_corrects += torch.sum(preds == labels.data).item()

        training_loss = training_loss / len(trainset)
        training_acc = training_corrects /len(trainset)
        print('\n Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(training_loss,training_acc))


        writer.add_scalars('Loss', {'Training loss': training_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Training accuracy': training_acc}, epoch + 1)


        writer.close()


        # Check best accuracy model ( but not the best on test )
        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())


        with open('TrainingAccuracy.txt','a') as fAcc:
            print('{:.4f} '.format(training_acc), file = fAcc)
        with open('TrainingLoss.txt','a') as fLoss:
            print('{:.4f} '.format(training_loss), file = fLoss)
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model, './Checkpoint/Resnet50_model-epoch-{:d}-train.pth'.format(epoch + 1))

        model = model.cuda(CUDA_DEVICES)
        model.eval()
        total_correct = 0
        total = 0
        class_correct = list(0. for i in enumerate(classes))
        class_total = list(0. for i in enumerate(classes))

        with torch.no_grad():
            for inputs, labels in tqdm(valid_dl):
                inputs = Variable(inputs.cuda(CUDA_DEVICES))
                labels = Variable(labels.cuda(CUDA_DEVICES))
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()


                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            for i, c in enumerate(classes):
              if(class_total[i]==0):
                print('Accuracy of %5s : %8.4f %%' % (
                c, 100 * 0))
              else:
                print('Accuracy of %5s : %8.4f %%' % (
                c, 100 * class_correct[i] / class_total[i]))

            # Accuracy
            print('\nAccuracy on the ALL val images: %.4f %%'
              % (100 * total_correct / total))

    # Save best training/valid accuracy model ( not the best on test )
    model.load_state_dict(best_model_params)
    best_model_name = './Checkpoint/model-{:.2f}-best_train_acc.pth'.format(best_acc)
    torch.save(model, best_model_name)
    print("Best model name : " + best_model_name)



if __name__ == '__main__':
    data_augmentation('./Dataset/curated_data/curated_data/1NonCOVID')
    data_augmentation('./Dataset/curated_data/curated_data/2COVID')
    train()