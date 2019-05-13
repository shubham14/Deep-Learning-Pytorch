# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:01:10 2019

@author: Shubham
"""

import torch
import torchtext
import numpy as np
import os, sys
from os.path import join as pjoin
import pandas as pd
from functools import reduce
import logging

data_dir = os.getcwd()
data_dir = os.path.join(data_dir, 'data')

logging.basicConfig(level=logging.DEBUG)

class DataLoader:
    def __init__(self, batch_size):
        self.data_dir = pjoin(os.getcwd(), 'data')
        self.batch_size = batch_size
        
    def load_data(self, mode):
        '''
        Function to extract the data and labels from the
        given text files
        '''
        file_name = mode + '.txt'
        if 'unlabelled' != mode:
            path = pjoin(self.data_dir, file_name)
            f = open(path, 'r').read().split('\n')
            d = list(map(lambda x: x[2:], f[:-1]))
            l = list(map(lambda x: int(x[0]), f[:-1]))
            m = max(list(map(lambda x: len(x.split()), d)))
            return d, l, m
        path = os.path.join(self.data_dir, file_name)
        d = open(path, 'r').read().split('\n')
        return d 

    def createVocab(self, d):
        '''
        Takes in the list of sentences processed from load_data
        '''
        logging.info("Creating Vocabulary from training data")
        words = list(map(lambda x: x.split(), d))
        words = [item for sublist in words for item in sublist]
        words = set(words)
        words.add(' ')
        vocab = dict(zip(words, np.arange(1, len(words) + 1)))
        vocab['<unk>'] = len(words) + 1
        vocab['<pad>'] = 0
        return vocab
    
    
    def padText(self, d, m, v):
        '''
        Pads data and creates a mapping from text to vocab 
        '''
        logging.info("Padding and encoding data")
        enc_data = np.zeros((len(d), m))
        d_padded = list(map(lambda x: x.split() + ['<pad>'] * (m - len(x.split())), d))
        for i in range(len(d)):
            for j in range(m):
                if d_padded[i][j] in v:
                    enc_data[i, j] = v[d_padded[i][j]]
                else:
                    enc_data[i, j] = v['<unk>']
        return enc_data
    
    def createTensor(self, data, labels, mode='train', batch_size=128):
        '''
        Creates a iterator over data for input to torch model 
        '''
        if mode == 'train':
            data = torch.Tensor(data)
            labels = torch.Tensor(labels)
            data_convert = torch.utils.data.TensorDataset(data, labels.view(-1, 1))
            data_loader = torch.utils.data.DataLoader(data_convert, batch_size=batch_size, shuffle=True)
            return data_loader
        elif mode == 'unlabelled':
            data = torch.Tensor(data)
            data_convert = torch.utils.data.TensorDataset(data)
            data_loader = torch.utils.data.DataLoader(data_convert, batch_size=batch_size, shuffle=True)
            return data_loader

    
if __name__ == "__main__":
    # main for testing functions for training part
    d = DataLoader(batch_size=64)
    text, labels, maxLen = d.load_data('train')
    v = d.createVocab(text)
    enc_data = d.padText(text, maxLen, v)
    data_loader = d.createTensor(enc_data, labels)
        