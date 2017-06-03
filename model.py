import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt

class ChunkSampler(sampler.Sampler):
    def __init__(self,num_samples,start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start,self.start+self.num_samples))

    def __len__(self):
        return self.num_samples

#total training samples = 50000
#size of samples is 1x32x32
NUM_TRAIN = 48000
NUM_VAL = 2000

batch_size = 64

cifar_train = dset.CIFAR10('./data',train=True,download=True, transform=T.ToTensor())
loader_train = DataLoader(cifar_train, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN,0))

cifar_val = dset.CIFAR10('./data',train=True,download=True, transform=T.ToTensor())
loader_val = DataLoader(cifar_val, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL,NUM_TRAIN))

cifar_test = dset.CIFAR10('./data',train=False,download=True, transform=T.ToTensor())
loader_test = DataLoader(cifar_test, batch_size=batch_size)

dtype = torch.FloatTensor #CPU datatype

print_every = 20

def reset(m):
    if hasattr(m,'reset_parameters'):
        m.reset_parameters()

class Flatten(nn.Module):
    def forward(self,x):
        N,C,H,W = x.size()
        return x.view(N,-1)


def train(model,loss_fn,optimizer,num_epochs):
    it_count = 0
    for epoch in range(num_epochs):
        acc_hist.append([it_count,check_accuracy(model,loader_val)])
        print('Starting epoch %d/%d' %(epoch+1,num_epochs))
        model.train()
        for t, (x,y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())
             
            scores = model(x_var)

            loss = loss_fn(scores,y_var)
            loss_hist.append([it_count,loss.data[0]])
            if((t+1)%print_every == 0):
                print('%d, loss = %.4f' % (t+1,loss.data[0]))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            it_count += 1

def check_accuracy(model,loader):
    if loader.dataset.train:
        print("Checking accuracy on validation set...")
    else:
        print("Cheching accuracy on test set...")
    num_correct = 0
    num_samples = 0
    model.eval() # put model into evaluation mode
    for x,y in loader:
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds==y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct)/num_samples
    print("Got %d/%d correct: %.2f" %(num_correct,num_samples,acc*100))
    return acc


conv = nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1), #32x32x32
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32,32,kernel_size=7,stride=1,padding=0), #32x26x26
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=0), #32x13x13
        Flatten(),
        nn.Linear(5408,2048),
        nn.Linear(2048,10)
        )
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(conv.parameters(),lr=1e-2)


loss_hist = []
acc_hist = []
conv.apply(reset)
train(conv,loss_fn,optimizer,num_epochs = 1)
check_accuracy(conv,loader_val)

loss_hist = np.array(loss_hist)
acc_hist = np.array(acc_hist)

plt.plot(loss_hist[:,0],loss_hist[:,1],'b-',acc_hist[:,0],acc_hist[:,1],'r-')
plt.show()
