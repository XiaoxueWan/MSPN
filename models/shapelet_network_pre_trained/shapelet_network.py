# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:43:06 2022

@author: Lenovo
"""
import torch
from torch import nn
from .shapeletsDistBlocks import ShapeletsDistBlocks

class LearningShapeletsModel(nn.Module):
    """
    """
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',ucr_dataset_name='comman',
                 to_cuda=True):
        super(LearningShapeletsModel, self).__init__()

        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure,ucr_dataset_name=ucr_dataset_name, to_cuda=to_cuda)
        self.linear = nn.Linear(self.num_shapelets, num_classes)
        self.Batch=nn.BatchNorm1d(self.num_shapelets)
        self.Batch1=nn.BatchNorm1d(num_classes)
        self.Sig=nn.Sigmoid()
        self.Re=nn.ReLU()

        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc'):
        """
        """
        m = self.shapelets_blocks(x)
        '''新增加的'''
        return m

    def transform(self, X):
        """
        """
        return self.shapelets_blocks(X)

    def get_shapelets(self):
        """
        """
        return self.shapelets_blocks.get_shapelets()

    def set_shapelet_weights(self, weights):
        """
        """
        start = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            end = start + num_shapelets
            self.set_shapelet_weights_of_block(i, weights[start:end, :, :shapelets_size])
            start = end

    def set_shapelet_weights_of_block(self, i, weights):
        """
        """
        self.shapelets_blocks.set_shapelet_weights_of_block(i, weights)

    def set_weights_of_shapelet(self, i, j, weights):
        """
        """
        self.shapelets_blocks.set_shapelet_weights_of_single_shapelet(i, j, weights)