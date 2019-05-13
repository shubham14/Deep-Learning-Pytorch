'''
Preprocessing the data and creating data loaders for input to the pytorch models
'''

import numpy as np
import re
import pickle as pkl
import pandas as pd
import torch.nn as nn
import spacy
from torchtext import data, vocab

nlp = spacy.load('en_core_web_sm')
MAX_CHARS = 20000

def tokenizer(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", 
        str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [x.text for x in nlp.tokenizer(comment) if x.text != " "]


def createVocab(data):
    words = list(map(lambda x: x.split(), data))
    max_len = max(list(map(lambda x: len(x), words)))
    words = [item for sublist in words for item in sublist]
    words = set(words)
    words.add(' ')
    vocab = dict(zip(words, np.arange(1, len(words) + 1)))
    vocab['unk'] = len(words) + 1
    return vocab, max_len

def createData(data_file, mode='labelled'):
    if mode == 'labelled':
        data_read = open(data_file, 'r').read()
        data_lines = data_read.split('\n')[:-1]
        
        # process the data and labels from the text file
        data1 = np.array(list(map(lambda x: x[2:], data_lines)))
        labels = np.array(list(map(lambda x: int(x[0]), data_lines)))
        
        print(data1.shape)
        # shuffling the data-label pairs for training
        rand_ind = np.random.permutation(len(data1))
        data1 = data1[rand_ind]
        labels = labels[rand_ind]
        vocab, maxLen = createVocab(data_lines)
        vocabSize = len(vocab.keys())
        return data1, labels, vocab, maxLen, vocabSize
    if mode != 'labelled':
        data = open('data/unlabelled.txt', 'r').read()
        data_lines = data.split('\n')
        return data_lines


def onehotEncode(data1, vocab1, vocabSize):
    n = np.zeros((len(data1), vocabSize + 1))
    for i in range(len(data1)):
        l = []
        l_split = data1[i].split()
        for word in l_split:
            if word in vocab1:
                l.append(vocab1[word])
            else:
                l.append(vocabSize)
        l = np.array(l)
        n[i,:][l] = 1            
    return n

# creating a dataLoader class for torchtext to do its thing 
class DataLoader:
    def __init__(self, vectors=None):
        self.vectors = vectors
    
    # function for loading pretrained vectors
    def load_vectors(self, vector_path, vector_name):
        vec = vocab.Vectors(vector_name, vector_path)
        self.vectors = vec
    
    def create_csv(self, data_file, name='train'):
        if name == 'unlabelled':
            with open(data_file, 'r') as f:
                  lines = f.readlines()
                  data_lines = [line.strip() for line in lines]
            df = pd.DataFrame(data_lines)
            csv_file = name + '.csv'
            df.to_csv(csv_file, header=['comment'], index=False)

        else:    
            data = open(data_file, 'r').read()
            data_lines = data.split('\n')[:-1]
            
            # process the data and labels from the text file
            data = np.array(list(map(lambda x: x[2:], data_lines)))
            labels = np.array(list(map(lambda x: int(x[0]), data_lines)))
            
            # shuffling the data-label pairs for training
            rand_ind = np.random.permutation(len(data))
            data = data[rand_ind]
            labels = labels[rand_ind]

            z = list(zip(list(data), list(labels)))
            df = pd.DataFrame(z)
            csv_file = name + '.csv'
            df.to_csv(csv_file, header=['comment', 'labels'], index=False)
        
    def get_dataset(self, tokenizer, lower=True, use_embeddings=False):
        '''
        For using GloVe embeddings use vectors=glove.6d
        '''
        if use_embeddings:
            self.load_vectors(r'C:\Users\Shubham\Desktop\deep-learning-course\Homeworks\Homework2\code', 'glove.6B.300d.txt')
        if self.vectors is not None:
            # pretrain vectors only supports all lower cases
            lower = True
        text = data.Field(
            sequential=True,
            tokenize=tokenizer,
            pad_first=True,
            lower=lower
        )
        train, val = data.TabularDataset.splits(
            path='', format='csv', skip_header=True,
            train='train.csv', validation='dev.csv',
            fields=[
                ('comment', text),
                ('labels', data.Field(
                    use_vocab=False, sequential=False)),
            ])
        test = data.TabularDataset(
            path='test.csv', format='csv', 
            skip_header=True,
            fields=[
                ('comment', text),
                ('labels', data.Field(
                    use_vocab=False, sequential=False)),
            ])
        test1 = data.TabularDataset(
            path='unlabelled.csv', format='csv', 
            skip_header=True,
            fields=[
                ('comment', text),
            ])
        text.build_vocab(train, val,
            min_freq=1,
            vectors=self.vectors)
        return train, val, test, test1, text
    
    def get_iterator(self, dataset, batch_size, train=True, shuffle=False, repeat=False):
        dataset_iter = data.BucketIterator(
            dataset, batch_size=batch_size, device=0,
            train=train, shuffle=shuffle, repeat=repeat,
            sort=False)
        return dataset_iter

if __name__ == "__main__":
    data_file = 'data/train.txt'
    data1, labels, vocab1, maxLen, vocabSize = createData(data_file)
    n = onehotEncode(data1, vocab1, vocabSize)
    
    dataloader = DataLoader()
    modes = ['train', 'test', 'dev', 'unlabelled']
    for mode in modes:
        print('%s done' %mode)
        data_file = 'data/' + mode + '.txt'
        dataloader.create_csv(data_file, name=mode) 

    train, val, test, test1, text = dataloader.get_dataset(tokenizer, use_embeddings=True)
    train_iter = dataloader.get_iterator(train, 64)