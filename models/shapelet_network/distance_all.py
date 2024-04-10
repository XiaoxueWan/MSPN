# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:52:14 2022

@author: Lenovo
"""
import torch
from torch import nn

class MinEuclideanDistBlock(nn.Module):
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
        super(MinEuclideanDistBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        #self.shapelets.retain_grad()

    def forward(self, x):
        """
        给x增加一个下采样的部分
        """
        # unfold time series to emulate sliding window
        x0 = x.unfold(2, self.shapelets_size, 1).contiguous()
        # calculate euclidean distance
        x = torch.cdist(x0, self.shapelets.clone(), p=2)

        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)
        # hard min compared to soft-min from the paper
        x, _ = torch.min(x, 3)
        
#        x_m = torch.Tensor()
#        x_original = x
#        for high in [1,1.5,2]:
#            for i in range(1,3):
#               # print(x_original.shape,'x.shape')
#                #每2-1个点取一个数,假如是1,就没变，假如是2就是隔一个点取一个数
#                x = x_original[:,:,::i]
#                x = x.unfold(2, self.shapelets_size, 1).contiguous()
#                #x = x * self.high
#               # print(self.high,'')
#               # self.shapelets = torch.mm(self.high ,self.shapelets)
#                #self.shapelets = self.high*self.shapelets
#                #print(x.shape,self.shapelets.shape,self.num_shapelets,'x.shape,self.shapelets.shape,self.num_shapelets*********')
#                x = torch.cdist(x, high*self.shapelets, p=2)
#                '''求和是因为有多个通道，所以把多个通道的值求和'''
#               # print(x.shape,'x = torch.cdist(x, self.shapelets, p=2)')
#                x, _ = torch.min(x, 2) #x:dim [num_timeseries,num_通道,num_shapelets]
#               # print(x.shape,'x, _ = torch.min(x, 2)')
#                
#                x_m = torch.cat([x_m,x],1)    # 把不同的下采样的结果当作不同的通道 ，所以是在第二个维度上叠加 x_m:dim [num_timeseries,num_通道+X_m,num_shapelets]
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)

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

#        if not list(weights.shape) == list(self.shapelets.shape):
#            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)}"
#                             f"compared to {list(weights.shape)}")

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


class MaxCosineSimilarityBlock(nn.Module):
    """
    Calculates the cosine similarity of a bunch of shapelets to a data set and performs global max-pooling.
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
        super(MaxCosineSimilarityBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x):
        """
        1) Unfold the data set 2) calculate norm of the data and the shapelets 3) calculate pair-wise dot-product
        4) sum over channels 5) perform a ReLU to ignore the negative values and 6) perform global max-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the cosine similarity for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        # normalize with l2 norm
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        n_dims = x.shape[1]
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3) / n_dims
        # ignore negative distances
        x = self.relu(x)
        x, _ = torch.max(x, 3)
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)

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
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)} "
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)

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
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)} "
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()


class MaxCrossCorrelationBlock(nn.Module):
    """
    Calculates the cross-correlation of a bunch of shapelets to a data set, implemented via convolution and
    performs global max-pooling.
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
    # TODO Why is this multiple time slower than the other two implementations?
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.cuda()

    def forward(self, x):
        """
        1) Apply 1D convolution 2) Apply global max-pooling
        @param x: the data set of time series
        @type x: array(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the most similar values for each pair of shapelet and time series instance
        @rtype: tensor(n_samples, num_shapelets)
        """
        x = self.shapelets(x)
        x, _ = torch.max(x, 2, keepdim=True)
        return x.transpose(2, 1)

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.weight.data

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

        if not list(weights.shape) == list(self.shapelets.weight.data.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape"
                             f"{list(self.shapelets.weight.data.shape)} compared to {list(weights.shape)}")

        self.shapelets.weight.data = weights

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
        if not list(weights.shape) == list(self.shapelets.weight.data[j, :].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape"
                             f"{list(self.shapelets.weight.data[j, :].shape)} compared to {list(weights.shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets.weight.data[j, :] = weights