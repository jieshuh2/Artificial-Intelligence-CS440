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
within this file and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



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

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn()
        self.lrate = lrate
        self.W1 = nn.Linear(in_size, 32)
        self.W2 = nn.Linear(32, out_size)
        self.model = nn.Sequential(
          self.W1,
          nn.ReLU(),
          self.W2,
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lrate)
        # raise NotImplementedError("You need to write this part!")

    def set_parameters(self, params):
        """ Sets the parameters of your network.

        @param params: a list of tensors containing all parameters of the network
        """
        # raise NotImplementedError("You need to write this part!")
        for i, param in enumerate(model.parameters()):
            model[i] = params[i]
    
    def get_parameters(self):
        """ Gets the parameters of your network.

        @return params: a list of tensors containing all parameters of the network
        """
        # raise NotImplementedError("You need to write this part!")
        return model.parameters()

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        pred = self.model(x)
        return pred

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        # raise NotImplementedError("You need to write this part!")
        self.zero_grad()
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

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    means = train_set.mean(dim=1, keepdim=True)
    stds = train_set.std(dim=1, keepdim=True)
    train_set = (train_set - means) / stds

    net = NeuralNet(0.13, nn.CrossEntropyLoss, 3072, 2);
    losses = np.ones(n_iter)
    
    index = [i for i in range(n_iter)]
    random.shuffle(index)

    for i in index:
        startindex = i * batch_size %len(train_set)
        loss = net.step(train_set[startindex: startindex + batch_size], train_labels[startindex:startindex + batch_size])
        losses[i] = loss

    devmean = dev_set.mean(dim=1, keepdim = True)
    devstd = dev_set.std(dim=1, keepdim = True)
    dev_set = (dev_set - devmean) /devstd

    yhat = np.ones(len(dev_set))
    for i,data in enumerate(dev_set):
        Fw = net.forward(data)
        yhat[i] = torch.argmax(Fw)
    return losses,yhat,net
