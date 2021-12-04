# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class1 = 0
class2 = 1

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn()
        self.lrate = lrate
        self.conv = nn.Sequential(
          nn.Conv2d(3,32,5),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          nn.Conv2d(32,64,5),
          nn.ReLU(),
          nn.MaxPool2d(2, 2)
        )

        self.model = nn.Sequential(
          nn.Linear(64*5*5, 120),
          nn.ReLU(),
          nn.Linear(120, 2),
        )
        self.optimizer = optim.SGD(self.parameters(), lr = self.lrate)


    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        x = (x - means) / stds
        x = x.view(len(x), 3,32,32)
        pred = self.conv(x)
        pred = pred.view(-1, 64*5*5)
        pred = self.model(pred)
        return pred

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        # self.zero_grad()
        self.optimizer.zero_grad()
        pred = self.forward(x)
        output = self.loss_fn(pred, y)
        output.backward()
        self.optimizer.step()
        return output.item()

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    net = NeuralNet(0.06, nn.CrossEntropyLoss, 3072, 2);
    losses = np.ones(n_iter)
    index = [i for i in range(n_iter)]
    random.shuffle(index)
    for i in index:
        startindex = i * batch_size %len(train_set)
        data = train_set[startindex: startindex + batch_size]
        # x =  data.view(batch_size, 3, 32, 32)
        loss = net.step(data, train_labels[startindex:startindex + batch_size])
        losses[i] = loss

    Fw = net.forward(dev_set)
    large = torch.argmax(Fw, dim = 1)
    yhat = large.numpy()
    # print(Fw.size())
    # for i in FW:
    #     yhat[i] = torch.argmax(i)
    return losses,yhat,net
