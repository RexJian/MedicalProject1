import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import time
import os
import torchvision.models as models
from tqdm import tqdm
from typing_extensions import Literal
from functools import reduce
import csv

def load_patients(csv_path, data_dir_path,mode):
    patients = {}
    with open(csv_path, encoding= 'unicode_escape') as csvFile :
        csvDictReader = csv.DictReader(csvFile)
        for row in csvDictReader:
            pid = row["Patient ID"]
            if patients.get(pid) is None:
                patients[pid] = []
            augmentation_file_name=row['File name'].replace('.png','_1.png')
            if mode=='train' and os.path.exists(os.path.join(data_dir_path,augmentation_file_name)) :
                patients[pid].append(os.path.join(data_dir_path, augmentation_file_name))
            else:
                print(mode+" :not exist"+augmentation_file_name)
            patients[pid].append(os.path.join(data_dir_path, row["File name"]))

    return [patient for patient in patients.values()]

def load_patients2(csv_path, data_dir_path):
    patients = {}
    with open(csv_path, encoding= 'unicode_escape') as csvFile : 
        csvDictReader = csv.DictReader(csvFile) 
        for row in csvDictReader:
            pid = row["Patient ID"]
            if patients.get(pid) is None:
                patients[pid] = []
            patients[pid].append(os.path.join(data_dir_path, row["File name"]))

    return [patient for patient in patients.values()]

def percent_list_slice(x, start=0., end=1.):
    return x[int(len(x)*start):int(len(x)*end)]

class CovidCT(Dataset):
    def __init__(self,
                 data_root,
                 mode: Literal["train", "valid", "test"] = "train",
                 transform=None):
        if mode == "train":
            start, end = 0.0, 0.7
        elif mode == "valid":
            start, end = 0.7, 0.8
        elif mode == "test":
            start, end = 0.8, 1.0

        normal_patients = load_patients(
            os.path.join(data_root, "meta_data_normal.csv"),
            os.path.join(data_root, "curated_data/curated_data/1NonCOVID"),
            mode)
        normal_patients = percent_list_slice(normal_patients, start, end)
        normal_file_paths = reduce(lambda a, b: a+b, normal_patients)
        
        covid_patients = load_patients(
            os.path.join(data_root, "meta_data_covid.csv"),
            os.path.join(data_root, "curated_data/curated_data/2COVID"),
            mode)
        covid_patients = percent_list_slice(covid_patients, start, end)
        covid_file_paths = reduce(lambda a, b: a+b, covid_patients)

        self.file_paths = normal_file_paths + covid_file_paths
        self.labels = [0]*len(normal_file_paths) + [1]*len(covid_file_paths)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = Image.open(self.file_paths[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]