# *- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:01:10 2019

1-D CNN for text classification

@author: Shubham
"""


import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_loader import *
import time
import logging

logging.basicConfig(level=logging.DEBUG)


class CNN(nn.Module):
    def __init__(self, inp_dim, embed_dim, output_dim, num_filters=128):
        super(CNN, self).__init__()
        self.inp_dim = inp_dim
        self.embedding = nn.Embedding(inp_dim, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, num_filters, 5)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_filters, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    # global avg-pooling
#    def forward(self, x):
#        x = self.embedding(x.long())
#        x_permute = x.permute(0, 2, 1)
#        x = self.conv1d(x_permute)
#        x_permute = x.permute(0, 2, 1)
#        x = x_permute.mean(dim=1)
#        x = self.relu(x)
#        x = self.linear(x)
#        out = self.sigmoid(x)
#        return out

    # global max-pooling 
    def forward(self, x):
        x = self.embedding(x.long())
        x_permute = x.permute(0, 2, 1)
        x = self.conv1d(x_permute)
        x = F.max_pool1d(x, kernel_size=x.size()[-1])
        x = self.relu(x)
        x = self.linear(x.permute(0, 2, 1))
        out = self.sigmoid(x)
        return out.squeeze(-1)
    
def accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    '''
    training drivers 
    '''
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.train()
    for text, labels in iterator:
        count += 1
        optimizer.zero_grad()
        preds = model(text)
        loss = criterion(preds, labels)
        acc = accuracy(preds, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if count % 100 == 0:
            print("Count {} : Training Accuracy {}".format(count, 
                                                  epoch_acc/len(iterator)))
    return epoch_loss/len(iterator) , epoch_acc/len(iterator)
        
def train_main(train_loader, vocab_dict, epochs):
    device = 'cpu'
    model = CNN(len(vocab_dict) + 1, 300, 1)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    for epoch in range(epochs):
        print("epoch {}/{}".format(epoch, epochs))        
        acc = train(model, train_loader, optimizer, criterion)
    return acc, model

def evaluate(data_loader, model, use_cuda=False):
    model.eval()
    pred = []
    test_labels = []
    for text, labels in data_loader:
        out = model(text)
        pred.append(out)
        test_labels.append(labels)
    pred = torch.stack(pred)
    test_labels = torch.stack(test_labels)
    acc = accuracy(pred.float(), test_labels.float())
    return float(acc) * 100.0, pred, test_labels 


def annotate_unlabelled(data_loader, model, use_cuda=False):
    model.eval()
    pred = []
    f = open('predictions_q1_1.txt', 'w')
    for text in data_loader:
        out = model(text)
        pred.append(out)
    pred = torch.stack(pred)
    pred = list(np.array(pred.detach()))
    for i in range(len(pred)):
        str1 = str(int(pred[i][0])) + ' ' + str(d[i])
        f.write(str1)
        f.write('\n')


if __name__ == "__main__":
    epochs = 10
    d = DataLoader(batch_size=64)
    text_tr, labels_tr, maxLen_tr = d.load_data('train')
    v = d.createVocab(text_tr)
    enc_data_tr = d.padText(text_tr, maxLen_tr, v)
    train_loader = d.createTensor(enc_data_tr, labels_tr)
    train_acc, model = train_main(train_loader, v, epochs)
    
    text_ts, labels_ts, maxLen_ts = d.load_data('test')
    enc_data_ts = d.padText(text_ts, maxLen_ts, v)  
    test_loader = d.createTensor(enc_data_ts, labels_ts, batch_size=1)
    test_acc, pred, test_labels = evaluate(test_loader, model)
       
    print("The training accuracy is:")
    print(train_acc)
    
    print("The testing accuracy is:")
    print(test_acc)
    
