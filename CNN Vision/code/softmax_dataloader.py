from softmax import SoftmaxClassifier
import numpy as np
from solver import Solver
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
from cnn import ConvNet

exists_1 = os.path.isfile('mnist_data.npy')
exists_2 = os.path.isfile('mnist_labels.npy')

if exists_1 and exists_2:
   print("Loading")
   X = np.load('mnist_data.npy')
   y = np.load('mnist_labels.npy')
else:
   print("Fetching")
   X , y = fetch_openml('mnist_784',version=1,return_X_y=True)
   y = y.astype(np.int32)
   np.save('mnist_data.npy',X)
   np.save('mnist_labels.npy',y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

X_train = X_train.reshape(-1, 1, 28, 28)
X_val = X_val.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
# N, D = X_train.shape
print (X_train.shape)
data = {'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
        }

model = ConvNet()
solver = Solver(model, data,
                 update_rule='sgd',
                 optim_config={
                   'learning_rate': 2e-3,
                 },
                 lr_decay=1,
                 num_epochs=1, batch_size=50,
                 print_every=2)

solver.train()

acc = solver.check_accuracy(X=X_test, y=y_test)
print(acc)
