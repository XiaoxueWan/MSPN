# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:52:14 2022

@author: Lenovo
"""
import torch
from torch import nn
from utils import normalize_data
     
class MinEuclideanDistBlockACS(nn.Module):
    """
    Calculates the euclidean distances of a bunch of shapelets to a data set and performs global min-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MinEuclideanDistBlockACS, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        
        high = torch.ones(self.num_shapelets)
        high = high.reshape(high.shape[0],1)
        self.high = nn.Parameter(high).contiguous()
       # self.high = self.high.reshape(high.shape[0],1)
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x):
        """
        1) Unfold the data set 2) calculate euclidean distance 3) sum over channels and 4) perform global min-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the euclidean for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        # calculate euclidean distance
        x = torch.cdist(x, self.shapelets, p=2)
        
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        '''求和是因为有多个通道，所以把多个通道的值求和'''
        #x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)
        # hard min compared to soft-min from the paper
        x, _ = torch.min(x, 2)
        x,__=torch.min(x,1)
        #print(x.shape,'x,__=torch.min(x,1)')
        x=x.reshape(x.shape[0],1,x.shape[1])
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)
    
    def get_high_parameters(self):
        return self.high

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        # transpose since internally we need shape (in_channels, num_shapelets, shapelets_size)
        weights = weights.transpose(1, 0)

        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)}"
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)
        self.shapelets.retain_grad()

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)}"
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()
        self.shapelets.retain_grad()
