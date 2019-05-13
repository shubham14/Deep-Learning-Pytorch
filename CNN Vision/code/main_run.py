from softmax import SoftmaxClassifier
from cnn import ConvNet
from logistic import LogisticClassifier
from svm import SVM
import numpy as np
from solver import Solver
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import pickle

def load_data(choice):
    if choice.lower() == 'softmax' or choice.lower() == 'convnet':
        exists_1 = os.path.isfile('mnist_data.npy')
        exists_2 = os.path.isfile('mnist_labels.npy')

        if exists_1 and exists_2:
            print("Loading")
            X = np.load('mnist_data.npy')
            y = np.load('mnist_labels.npy')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

            print (X_train.shape)
            data = {'X_train': X_train, 'y_train': y_train,
                    'X_val': X_val, 'y_val': y_val,
                    'X_test': X_test, 'y_test': y_test
                    }

        else:
            print("Fetching")
            X , y = fetch_openml('mnist_784',version=1,return_X_y=True)
            y = y.astype(np.int32)
            np.save('mnist_data.npy',X)
            np.save('mnist_labels.npy',y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

            print (X_train.shape)
            data = {'X_train': X_train, 'y_train': y_train,
                    'X_val': X_val, 'y_val': y_val,
                    'X_test': X_test, 'y_test': y_test
                    }
        return data

    with open('data.pkl', 'rb') as f:
        X,y = pickle.load(f, encoding='latin1')

    x_train , y_train = X[:500] , y[:500]
    x_val , y_val = X[500:750] , y[500:750]
    x_test , y_test = X[750:] , y[750:]

    data = {'X_train': x_train, 'y_train': y_train,
        'X_val': x_val, 'y_val': y_val,
        'X_test': x_test, 'y_test': y_test
        }

    return data


data = load_data(choice='multi')
print(data['X_train'].shape)
model = SoftmaxClassifier(hidden_dim=50, reg=0)

solver = Solver(model, data,
                 update_rule='sgd',
                 optim_config={
                   'learning_rate': 2e-3,
                 },
                 lr_decay=1,
                 num_epochs=1, batch_size=50,
                 print_every=2)
        
solver.train()

acc = solver.check_accuracy(X=data['X_test'], y=data['y_test'])
print(acc)
