import torch
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from Data_Processing import CovidCT
import os

CUDA_DEVICES = 0
PATH_TO_WEIGHTS = r'C:/Users/dsp523/Desktop/MIP_Project1/Checkpoint/model-0.89-best_train_acc.pth' # Your model name
#PATH_TO_WEIGHTS = r'C:/Users/dsp523/Desktop/MIP_Project1/Checkpoint/model-epoch-20-train.pth' # Your model name


def test():
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    testset = CovidCT(r"C:/Users/dsp523/Desktop/MIP_Project1/", "test", data_transform)
    # testset = CovidCT(f"{os.getcwd()+'./Dataset/'}", "test", data_transform)

    test_dl = DataLoader(testset, batch_size=7, shuffle=False, pin_memory=True, num_workers=4)
    classes = ['1NonCOVID','2COVID']

    # Load model
    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()
    
    total_correct = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))

    with torch.no_grad():
        for inputs, labels in tqdm(test_dl):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            
            # batch size
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i, c in enumerate(classes):
        print('Accuracy of %5s : %8.4f %%' % (
        c, 100 * class_correct[i] / class_total[i]))

    # Accuracy
    print('\nAccuracy on the ALL test images: %.4f %%'
      % (100 * total_correct / total))


if __name__ == '__main__':
    test()
