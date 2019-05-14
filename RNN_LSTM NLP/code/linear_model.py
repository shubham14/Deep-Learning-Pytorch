import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess_data import *
import gensim
import torch.optim as optim
from torchtext import data, vocab 

np.random.seed(42)

class LinearNet(nn.Module):
    def __init__(self, inp_dim, output_dim):
        super(LinearNet, self).__init__()
        self.inp_dim = inp_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.inp_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inp):
        out = self.linear(inp)
        out = self.sigmoid(out)
        return out

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
            print(epoch_acc/len(iterator))
    return epoch_loss/len(iterator) , epoch_acc/len(iterator)


def train_main(epochs):
    device = 'cpu'
    data_file = 'data/train.txt'
    data1, labels, vocab1, maxLen, vocabSize = createData(data_file)
    n = torch.Tensor(onehotEncode(data1, vocab1, vocabSize))
    labels = torch.Tensor(labels)
    train_convert = torch.utils.data.TensorDataset(n, labels.view(-1, 1))
    train_loader = torch.utils.data.DataLoader(train_convert, batch_size=64, shuffle=False)
    model = LinearNet(n.size(1), 1)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    for epoch in range(epochs):
        print("epoch {}/{}".format(epoch, epochs))        
        acc = train(model, train_loader, optimizer, criterion)
    return acc, model, vocab1


def test_main(model, vocab1):
    device = 'cpu'
    data_file = 'data/test.txt'
    data1, labels, _, _, _ = createData(data_file)
    dataloader = DataLoader()
    n = torch.Tensor(onehotEncode(data1, vocab1, len(vocab1)))
    labels = torch.Tensor(labels)
    print("testing here")
    test_convert = torch.utils.data.TensorDataset(n, labels.view(-1, 1))
    test_loader = torch.utils.data.DataLoader(test_convert, batch_size=1, shuffle=False)
    print(test_loader)
    pred = []
    test_labels = []
    for text, labels in test_loader:
        out = model(text)
        pred.append(out)
        test_labels.append(labels)
    pred = torch.stack(pred)
    test_labels = torch.stack(test_labels)
    acc = accuracy(pred.float(), test_labels.float())
    return float(acc) * 100.0    

def annotateResults(data_file, vocab1):
    d = createData(data_file, mode='unlabelled')
    d1 = d[:-2]
    f = open('predictions_q1_1.txt', 'w')
    n = torch.Tensor(onehotEncode(d1, vocab1, len(vocab1)))
    test_convert = torch.utils.data.TensorDataset(n)
    test_loader = torch.utils.data.DataLoader(test_convert, batch_size=1, shuffle=False)
    pred = []
    for text in test_loader:
        out = model(text[0])
        pred.append(out)
    pred = torch.round(torch.stack(pred))
    pred = list(np.array(pred.detach()))
    for i in range(len(pred)):
        str1 = str(int(pred[i][0])) + ' ' + str(d[i])
        f.write(str1)
        f.write('\n')

    
if __name__ == "__main__":
    epochs = 20
    train_acc, model, vocab1 = train_main(epochs)
    test_acc = test_main(model, vocab1)
    print('The training accuracy is')
    print(train_acc)
    print('The testing accuracy is')
    print(test_acc)
    annotateResults(model, vocab1)