"""
@author: Czander

CNN -- Activation: ReLU, Loss function: Cross Entropy, Optimizer: SGD

(1 epoch, batch size: 1)
Training time: ~3 mins
Accuracy: 10.28?

(1 epoch, batch size: 500)
Training time: ~14 secs
Accuracy: 95.44

(1 epoch, batch size: 100)
Training time: ~11 secs
Accuracy: 97.13
(2 epochs, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
Accuracy: 98.07, 98.39, 98.46, 98.47, (98.44), 98.53, 98.55, 98.58, 
            (98.54), 98.72, 98.76, (98.64), *98.82*, (98.74), ((98.62)), (98.69), 
            ((98.64)), ((98.53)), ((98.39)), (98.65)
"""
import time
from tensorflow import keras

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Creating datasets and dataloaders (with batch size) for torch
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float), 
                           torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float), 
                          torch.tensor(y_test, dtype=torch.long))

batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Step by Step Process of CNN -- (ReLU activations)
# Conv1 -> Max Pool -> Conv2 -> Max Pool -> FC1 -> FC2 -> FC3
for i, (x, labels) in enumerate(train_loader):
    x = x.view([batch_size, 1, 28, 28])     # specifying 1 channel image (no rgb)
    if i==0: break

conv1 = nn.Conv2d(1, 10, 3)
conv2 = nn.Conv2d(10, 20, 4)
fc1 = nn.Linear(20 * 5 * 5, batch_size)
fc2 = nn.Linear(batch_size, 128)
fc3 = nn.Linear(128, 10)

y = conv1(x)                    # 1st conv layer
print(y.size())

z = F.relu(y)                   # 1st ReLU activation
print(z.size())

x2 = F.max_pool2d(z, (2, 2))    # 1st max pooling layer (2d)
print(x2.size())

y2 = conv2(x2)                  # 2nd conv layer
print(y2.size())

z2 = F.relu(y2)                 # 2nd ReLU activation
print(z2.size())

x3 = F.max_pool2d(z2, (2, 2))   # 2nd max pooling layer (2d)
print(x3.size())

x3 = x3.view(batch_size, -1)                # Flattening
print(x3.size())

x4 = F.relu(fc1(x3))            # 1st Fully Connected Layer
print(x4.size())

x5 = F.relu(fc2(x4))            # 2nd Fully Connected Layer
print(x5.size())

x6 = F.relu(fc3(x5))            # 3rd Fully Connected Layer (output: 10 classes)
print(x6.size())

pred = torch.max(x6.data, 1)[1]     # Calculating results
count = 0
for j,item in enumerate(pred):
    count += int(item == labels[j])         # Obviously results will suck
print(count, "/", len(labels), "correct")   # with only training on one batch of 100


# Building CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # input [batch_size x 1 x 28 x 28], 10 output nodes, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 4)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20 * 5 * 5, self.batch_size)
        self.fc2 = nn.Linear(self.batch_size, 128)
        self.fc3 = nn.Linear(128, 10)     # 10 output nodes
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# For training model on data
def train_nn(net, train_loader, batch_size=batch_size):
    start = time.time()
    for i,(feats, labels) in enumerate(train_loader):
        feats = feats.view([batch_size, 1, 28, 28])  
    
        optimizer.zero_grad()

        output = net(feats)
    
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if i*batch_size%100==0: 
            print(i*batch_size, "/", len(train_loader)*batch_size)

    print(round((time.time() - start), 2), "s")

# For testing model with test dataset
def test_nn(net, test_loader, batch_size=batch_size):
    count = 0
    for i,(test, labels) in enumerate(test_loader):
        test = test.view([batch_size, 1, 28, 28])

        test_out = net(test)
        pred = torch.max(test_out, 1)[1]
        for j,item in enumerate(pred):
            count += int(item == labels[j])
    
        if i*batch_size%100==0:
            print(i*batch_size, "/", len(test_loader)*batch_size)
            print(count, "/", i*batch_size)
    
    print("Accuracy: ", (count/len(x_test))*100)

# Initializing model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_nn(net, train_loader)
test_nn(net, test_loader)

# Saving/loading model
PATH = './torch_conv100.nn'
torch.save(net.state_dict(), PATH)
#net = Net()
#net.load_state_dict(torch.load(PATH))