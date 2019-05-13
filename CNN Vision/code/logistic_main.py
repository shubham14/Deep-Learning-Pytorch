import numpy as np
from solver import Solver
from logistic import LogisticClassifier
from svm import SVM
from cnn import ConvNet
import pickle
from softmax import SoftmaxClassifier

with open('data.pkl', 'rb') as f:
   X,y = pickle.load(f, encoding='latin1')

x_train , y_train = X[:500] , y[:500]
x_val , y_val = X[500:750] , y[500:750]
x_test , y_test = X[750:] , y[750:]

data = {'X_train': x_train, 'y_train': y_train,
       'X_val': x_val, 'y_val': y_val,
       'X_test': x_test, 'y_test': y_test
       }

N,D = x_train.shape

model = LogisticClassifier(input_dim=D, hidden_dim=100, reg=0.05)
solver = Solver(model, data,
                 update_rule='sgd',
                 optim_config={
                   'learning_rate': 5e-2,
                 },
                 lr_decay=1,
                 num_epochs=1000, batch_size=20,
                 print_every=200)
solver.train()

acc = solver.check_accuracy(x_test, y_test)
print(acc)