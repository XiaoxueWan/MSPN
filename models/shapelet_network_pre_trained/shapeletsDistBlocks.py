# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:44:27 2022

@author: Lenovo
"""
from collections import OrderedDict
from torch import nn
import torch
import numpy as np
from .distanceACS import MinEuclideanDistBlockACS
from .distance_all import MinEuclideanDistBlock,MaxCosineSimilarityBlock,MaxCrossCorrelationBlock

class ShapeletsDistBlocks(nn.Module):
    """
    因为ACS是多变量时间序列信号，所以需要根据数据集是否是阳极电流信号数据集来选择计算x与shapelet距离的方式。
    MinEuclideanDistBlock是一般的公共数据集的计算方法
    MinEuclideanDistBlockACS是阳极电流信号数据集的计算方法
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean',ucr_dataset_name='comman',to_cuda=True):
        super(ShapeletsDistBlocks, self).__init__()
        self.to_cuda = to_cuda
       # self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.shapelets_size_and_len = shapelets_size_and_len
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        if dist_measure == 'euclidean':
            if ucr_dataset_name=='ACS':
                self.blocks = nn.ModuleList(
                    [MinEuclideanDistBlockACS(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                           in_channels=in_channels, to_cuda=self.to_cuda)
                     for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
            else:
                self.blocks = nn.ModuleList(
                    [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                           in_channels=in_channels, to_cuda=self.to_cuda)
                     for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

    def forward(self, x):
        """
        Calculate the distances of each shapelet block to the time series data x and concatenate the results.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: a distance matrix containing the distances of each shapelet to the time series data
        @rtype: tensor(float) of shape
        """
        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        for block in self.blocks:
            #print('**********',out.shape,block.shapelets.shape,block.shapelets_size,'out.shapeshapelets.shape')
            out = torch.cat((out, block(x)), dim=1)
        return out

    def get_blocks(self):
        """
        @return: the list of shapelet blocks
        @rtype: nn.ModuleList
        """
        return self.blocks

    def get_block(self, i):
        """
        Get a specific shapelet block. The blocks are ordered (ascending) according to the shapelet lengths.
        @param i: the index of the block to fetch
        @type i: int
        @return: return shapelet block i
        @rtype: nn.Module, either
        """
        return self.blocks[i]

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of the shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.blocks[i].set_shapelet_weights(weights)

    def get_shapelets_of_block(self, i):
        """
        Return the shapelet of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @return: the weights of the shapelet block
        @rtype: tensor(float) of shape (in_channels, num_shapelets, shapelets_size)
        """
        return self.blocks[i].get_shapelets()

    def get_shapelet(self, i, j):
        """
        Return the shapelet at index j of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @return: return the weights of the shapelet
        @rtype: tensor(float) of shape
        """
        shapelet_weights = self.blocks[i].get_shapelets()
        return shapelet_weights[j, :]

    def set_shapelet_weights_of_single_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the new weights for the shapelet
        @type weights: array-like of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.blocks[i].set_weights_of_single_shapelet(j, weights)

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        max_shapelet_len = max(self.shapelets_size_and_len.keys())
        num_total_shapelets = sum(self.shapelets_size_and_len.values())
        shapelets = torch.Tensor(num_total_shapelets, self.in_channels, max_shapelet_len)
        shapelets[:] = np.nan
        start = 0
        for block in self.blocks:
            shapelets_block = block.get_shapelets()
            end = start + block.num_shapelets
            shapelets[start:end, :, :block.shapelets_size] = shapelets_block
            start += block.num_shapelets
        return shapelets
