# Usage
# python train_efficientnet.py --dataset data/mask_df.pickle --output models/v1_torch.pt
from __future__ import print_function, division

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import time
import os
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from barbar import Bar

from dataset import MaskDataset
from efficientnet_pytorch import EfficientNet
RANDOMSTATE = 0

class TransferLearningModel(nn.Module):
    @staticmethod
    def efficientnetmodel(version):
        basemodel = EfficientNet.from_pretrained(version)

        # Freeze model weights
        for param in basemodel.parameters():
            param.requires_grad = False
        num_ftrs = basemodel._fc.in_features

        basemodel._fc = nn.Linear(num_ftrs, 1)

        return basemodel

    @staticmethod
    def inception_v3():
        inception = models.inception_v3(pretrained=True)

        # Freeze model weights
        for param in inception.parameters():
            param.requires_grad = False

        num_ftrs = inception.fc.in_features
        inception.fc = nn.Linear(num_ftrs, 2)

        return inception

    @staticmethod
    def resnet18():
        resnet = models.resnet18(pretrained=True)
        num_ftrs = resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        resnet.fc = nn.Linear(num_ftrs, 1)

        return resnet
        

def prepare_data(maskDFPath):
    maskDF = pd.read_pickle(maskDFPath)
    train, validate = train_test_split(maskDF, test_size=0.15, random_state=RANDOMSTATE,
                                        stratify=maskDF['mask'])
    trainDF = MaskDataset(train)
    validateDF = MaskDataset(validate)
    return trainDF, validateDF

def load_ckpt(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])

    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # initialize valid_loss_min from checkpoint to valid_loss_min
    #valid_loss_min = checkpoint['valid_loss_min']

    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch']

def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint

def train_model(model, 
                start_epoch, 
                criterion, 
                optimizer, 
                #scheduler, 
                num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, start_epoch+num_epochs):
        print('Epoch {}/{}'.format(epoch, start_epoch+num_epochs-1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx,(inputs, labels) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = torch.as_tensor(labels, dtype = torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    m = nn.Sigmoid()

                    preds = torch.sigmoid(outputs)
                    preds = preds>0.5
    
                    loss = criterion(m(outputs), labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #if phase == 'train':
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_checkpoint(state={   
                                    'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'best_accuracy': best_acc,
                                    'optimizer_state_dict':optimizer.state_dict()
                                },filename='checkpoints/ckpt_epoch_{}.pt'.format(epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, optimizer, epoch_loss

if __name__=='__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-m", "--model", type=str,
        help="path to output face mask detector model")
    ap.add_argument("-o", "--output", type=str, default="models/v2.model",
        help="Model save path")
    args = vars(ap.parse_args())

    EPOCHS = 50
    start_epoch = 1
    PATH = args['output']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    trainDF, validateDF = prepare_data(maskDFPath=args['dataset'])
    dataloaders = {'train': DataLoader(trainDF, batch_size=32, shuffle=True, num_workers=10) ,
                    'val':DataLoader(validateDF, batch_size=32, num_workers=10)
                    }

    dataset_sizes = {'train': len(trainDF),'val':len(validateDF)}

    model = TransferLearningModel.resnet18()
    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCELoss()
    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if args['model'] != None:
        # load the saved checkpoint
        model, optimizer, start_epoch = load_ckpt(args['model'], model, optimizer)
        print('[INFO] Checkpoint Loaded')

    model, optimizer, loss = train_model(model=model, 
                        start_epoch=start_epoch,
                        criterion=criterion, 
                        optimizer=optimizer, 
                        #scheduler=exp_lr_scheduler,
                        num_epochs=EPOCHS)

    print('[INFO] Saving Model...')
    torch.save({
                'epoch': EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)



