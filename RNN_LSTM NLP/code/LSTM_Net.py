# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:49:04 2019

@author: Shubham
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess_data import *
import gensim
import torch.optim as optim
from torchtext import data, vocab 

np.random.seed(42)

class LSTM_Net(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, weights=None):
        super(LSTM_Net, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        if weights is not None:
            self.embedding.weight.data.copy_(weights)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        embed_out = self.embedding(inp)
        out, (hidden1, hidden2) = self.lstm(embed_out)
        out = self.sigmoid(self.linear(hidden1.squeeze(0)))
        return out.squeeze(-1)
    
def accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.train()
    for batch in iterator:
        count += 1
#        text, labels = iter_batch[0], iter_batch[1]
        batch.labels = batch.labels.float()
        optimizer.zero_grad()
        preds = model(batch.comment)
        loss = criterion(preds, batch.labels)
        acc = accuracy(preds, batch.labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if count % 100 == 0:
            print(epoch_acc/len(iterator))
    return epoch_loss/len(iterator) , epoch_acc/len(iterator), preds

def train_main(epochs):
    device = 'cpu'
    data_file = 'data/train.txt'
    data1, labels, vocab1, maxLen, vocabSize = createData(data_file)
    dataloader = DataLoader()
    modes = ['train', 'test', 'dev']
    for mode in modes:
        print('%s done' %mode)
        data_file = 'data/' + mode + '.txt'
        dataloader.create_csv(data_file, name=mode) 
    
    train_set, val_set, test_set, unlabelled_set, text = dataloader.get_dataset(tokenizer, use_embeddings=True)
    model = LSTM_Net(len(text.vocab), 300, 100, 1, weights=train_set.fields['comment'].vocab.vectors)
#    model = LSTM_Net(len(text.vocab), 300, 100, 1, weights=None)
    model = model.to(device)
    print("training here")
    train_loader = dataloader.get_iterator(train_set, 128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    for epoch in range(epochs):
        print("epoch {}/{}".format(epoch,  epochs))
        loss, acc, pred = train(model, train_loader, optimizer, criterion)
    return float(acc) * 100.0, model, vocab1, unlabelled_set
    
    
def test_main(model):
    device = 'cpu'
    data_file = 'data/test.txt'
    data1, labels, vocab1, maxLen, vocabSize = createData(data_file)
    dataloader = DataLoader()
    modes = ['train', 'test', 'dev']
    for mode in modes:
        print('%s done' %mode)
        data_file = 'data/' + mode + '.txt'
        dataloader.create_csv(data_file, name=mode) 
    train_set, val_set, test_set, unlabelled_set, text = dataloader.get_dataset(tokenizer, use_embeddings=True)
    print("testing here")
    test_loader = dataloader.get_iterator(test_set, 1, train=False)
    print(test_loader)
    pred = []
    test_labels = []
    for batch in test_loader:
        out = model(batch.comment)
        pred.append(out)
        test_labels.append(batch.labels)
    pred = torch.stack(pred)
    test_labels = torch.stack(test_labels)
    acc = accuracy(pred.float(), test_labels.float())
    
    return float(acc) * 100.0
        
def annotateResults(model, data_file, vocab1, unlabelled_set):
    d = createData(data_file, mode='unlabelled')
    f = open('predictions_q5.txt', 'w')
    pred = []
    dataloader = DataLoader()
    test_loader = dataloader.get_iterator(unlabelled_set, 1, train=False, shuffle=False)
    for text in test_loader:
        try:
            out = model(text.comment)
            pred.append(out)
        except:
            RuntimeError
    pred = torch.round(torch.stack(pred))
    pred = list(np.array(pred.detach()))
    for i in range(len(pred)):
        try:
            str1 = str(int(pred[i][0])) + '  ' + str(d[i])
            f.write(str1)
            f.write('\n')
        except:
            ValueError

if __name__ == "__main__":
    epochs = 20
    train_acc, model, vocab1, unlabelled_set = train_main(epochs)
    test_acc = test_main(model)
    print('The training accuracy is')
    print(train_acc)
    print('The testing accuracy is')
    print(test_acc)
    annotateResults(model, 'unlabelled.txt', vocab1, unlabelled_set)