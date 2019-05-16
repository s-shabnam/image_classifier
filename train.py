'''Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu'''
# call me by python train.py  --arch="vgg13"

import argparse
import myNetwork as mn
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='Trains a model for prediction')

parser.add_argument('--gpu', 
                    action = "store_true", 
                    dest = 'deviceIsGPU', 
                    default = False, 
                    help ='True if you want use CPU for training')

parser.add_argument('--learning_rate', 
                    action = "store", 
                    dest="learning_rate", 
                    default = 0.01,
                    type = float,
                    help ='Used to set desired learning rate for the model training')

parser.add_argument('--epochs', 
                    action = "store", 
                    dest = "epochs", 
                    default = 1,  
                    type = int, 
                    help ='Used to set the epochs number for the model training')

parser.add_argument('--hidden_units', 
                    action = "store", 
                    dest = "hidden_units", 
                    default = 512,  
                    type = int,
                    help = 'Used to set the hidden units number for the model training')

parser.add_argument('data_dir', 
                    action = "store", 
                    default = 'flowers',
                    help = 'Used to set the  data directory for the model training')

parser.add_argument('--save_dir', 
                    action = "store", 
                    dest = "save_dir", 
                    default = './', 
                    help = 'Used to set the  save directory to save the trained model')

parser.add_argument('--arch', 
                    action = "store", 
                    dest = "architecture", 
                    default = 'vgg19', 
                    help = 'Used to set the  pretrained model  architecture')

parser.add_argument('--version', action = 'version',
                    version = '%(prog)s 1.0')

input_args = parser.parse_args()
print(input_args)

print('Step 0: Begin processing...')

print('Step 1: Set test, validation and train data directories..')
data_sets = ['train', 'valid', 'test']
data_directories = mn.set_directories(input_args.data_dir)
print('Step 1: Done')

print('Step 2: Transform data...')
data_loaders, dataset_sizes, image_datasets = mn.transform_data(data_directories, data_sets)
print('Step 2: Done')

print('Step 3: Define model architecture...')
model, criterion, optimizer = mn.set_architecture(input_args.architecture, input_args.learning_rate)
print('Step 3: Done')

device = 'cuda'
if input_args.deviceIsGPU:
    device = 'gpu' 

print('Step 4: Train the selected model {} with input parameters: epochs = {} and device = {} ...'.format(input_args.architecture, input_args.epochs, device))
trained_model = mn.train_model(data_loaders, model, criterion, optimizer, input_args.epochs, device)
print('Step 4: Done')

print('Step 5: Check prediction arruracy...')
mn.check_accuracy(trained_model, data_loaders['test'], device)
print('Step 5: Done')

print('Step 6: Save trained model...')
mn.save_model(image_datasets, trained_model, input_args.save_dir)
print('Step 6: Done')




