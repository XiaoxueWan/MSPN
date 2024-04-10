# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:41:35 2022

@author: Lenovo
"""
import sys 
sys.path.append("..") 

import torch
import types
import warnings
import numpy as np

from torch import tensor
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from .shapelet_network.shapelet_network import LearningShapeletsModel
from utils import  visualize_2D

class LearningShapelets:
    """
  
    """
    def __init__(self, 
                 shapelets_size_and_len, 
                 num_classes, 
                 in_channels=1, 
                 dist_measure='euclidean',
                 ucr_dataset_name='comman', 
                 verbose=0, 
                 to_cuda=False, 
                 k=0, 
                 l1=0.0, 
                 l2=0.0
                 ):
        
        #model set 里面存放所有类别的模型
        self.model = LearningShapeletsModel(
                                            shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=1, 
                                            num_classes=num_classes, 
                                            dist_measure=dist_measure,
                                            ucr_dataset_name=ucr_dataset_name,
                                            to_cuda=to_cuda
                                            ) 
        
        
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.shapelets_size_and_len = shapelets_size_and_len
        self.verbose = verbose
        self.optimizer = None

        if not all([k == 0, l1 == 0.0, l2 == 0.0]) and not all([k > 0, l1 > 0.0]):
            raise ValueError("For using the regularizer, the parameters 'k' and 'l1' must be greater than zero."
                             " Otherwise 'k', 'l1', and 'l2' must all be set to zero.")
        self.k = k
        self.l1 = l1
        self.l2 = l2
        
        # add a variable to indicate if regularization shall be used, just used to make code more readable
        self.use_regularizer = True if k > 0 and l1 > 0.0 else False

    def set_optimizer(self, optimizer):
        """
        """
        self.optimizer = optimizer

    def set_shapelet_weights(self, weights):
        """
        """
        self.model.set_shapelet_weights(weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")

    def set_shapelet_weights_of_block(self, i, weights):
        """
        """
        self.model.set_shapelet_weights_of_block(i, weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")
        
        
    def get_prototype(self, D, y0):
        '''
           新来一个任务的数据，首选初始化计算当前任务对应的特征原型.
           Input: 
                  D:  特征矩阵
                       类型：tensor 
                       尺寸： [num_samples, num_view, num_shapelets] 
                   y: X对应的标签
                      类型： tensor 
                      [1,2,3]
           Output:
               Prototypes： 原型集 
                      类型： tensor 
                      {1:[],2:[],3:[]} 
                      R:原型数*num_shapelets
        '''
        Prototypes_current_task_multi_view = [] 
        y0 = y0.unsqueeze(1)
        y0 = y0.float()
        for i in range(3): #遍历每个视图
            D_y = torch.cat((D[:,i,:],y0),1) #将D拼上y 
            y = torch.squeeze(y0)
            y_num = y.numpy()
            Prototypes_current_task = dict.fromkeys(list(set(y_num)))  #新建一个空的字典，以y的值作为键
            for i in list(set(y_num)):
               mask = torch.where(D_y[:,-1]==i)
               D_y = D_y.float()
               D_y_class = torch.mean(D_y[mask][:,:-1],0)
               if Prototypes_current_task[i] == None:
                   Prototypes_current_task[i] = D_y_class
            Prototypes_current_task_multi_view.append(Prototypes_current_task)
        return Prototypes_current_task_multi_view

    def update(self, x, y, Prototypes):
        """
           变成增量的模型更新方式
        """
        type_prototypes = type(Prototypes[0])
        D = self.model(x)
        Prototypes_current_task = self.get_prototype(D,y)
        
        Prototypeslist = []
        for i in range(3): #更新每个视图
            if type_prototypes==torch.Tensor:  #假如Prototypes 是一个tensor，就对它进行转换，转成字典
                Prototypesdict = dict.fromkeys(list(range(len(Prototypes))))
                for j, val in Prototypesdict.items():#'再把tensor_Prototype 变为字典Prototype'
                    Prototypesdict[i] = Prototypes[i]
            else:
                Prototypesdict = Prototypes[i]
            Prototypesdict.update(Prototypes_current_task[i])  #除了第一次任务，所有原型为当前原型加上历史原型
            Prototypeslist.append(Prototypesdict)
        return Prototypeslist

    def fit(self, X, Y, num_task, Prototypes, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        """

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = tensor(X, dtype=torch.float).contiguous()
        if not isinstance(Y, torch.Tensor):
            Y = tensor(Y, dtype=torch.long).contiguous()
            
        if self.to_cuda:
            X = X.cuda()
            Y = Y.cuda()
        if Y.min()<0:
            Y=torch.sub(Y,Y.min())

        train_ds = TensorDataset(X, Y)
        train_dl = DataLoader(train_ds, batch_size=len(X), shuffle=shuffle, drop_last=drop_last)
        
        # set model in train mode
        self.model.train()

        losses_ce = []
        losses_dist = []
        losses_sim = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        for _ in progress_bar:
            for j, (x, y) in enumerate(train_dl):
                # check if training should be done with regularizer
                #print(Prototypes,'Prototypes********before')
                #print(y,'*****************y***************')
                Prototypes = self.update(x, y, Prototypes)
                #print(Prototypes,'Prototypes********letter')
                losses_ce.append(current_loss_ce)
               
        return Prototypes if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
        losses_ce, losses_dist) 

    def transform(self, X):
        """
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if self.to_cuda:
            X = X.cuda()

        with torch.no_grad():
            shapelet_transform = self.model.transform(X)
        return shapelet_transform.squeeze().cpu().detach().numpy()

    def fit_transform(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        """
        self.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return self.transform(X)
    
    def predict_by_prototypes(self, Pre_D, Prototypes):
        Y_list = np.array([list(range(len(Pre_D)))])
       # print(Y_list, '**********')
        for i in range(3): #遍历每个原型
            distance_proto_D = torch.tensor([])
            for classes,Prototype in  Prototypes[i].items():
                cos = torch.cosine_similarity(Pre_D[:,i,:],torch.tensor(Prototype),dim=-1)
                cos = cos.unsqueeze(1)
                distance_proto_D = torch.cat((distance_proto_D,cos),dim=1)
            Y = torch.argmax(distance_proto_D,dim=1).numpy()
            Y = Y.reshape((1,-1))
            Y_list = np.concatenate((Y_list,Y),axis=0)
        Y_list = Y_list[1:,:]
        #print(Y_list,'Y_list**********')
           # visualize_2D(Pre_D[:,i,:],torch.argmax(distance_proto_D,dim=1),'view'+str(i))
       # print(Y_list.shape,'&&&&&&&&&&&&&&Y_list')
       # print(Y_list[2],'&&&&&&&&&&&&&&Y_list[2')
        Y_list_last = []
        for j in range(len(Y_list[0])):
            Y_list_last.append(Counter(Y_list[:,j]).most_common(1)[0][0])
        return torch.Tensor(Y_list_last)
        #return torch.Tensor(Y_list[2])
    
    def predict_by_prototypes_limit(self, Pre_D, Prototypes):
       # print(Prototypes)
        Y_list = np.array([list(range(len(Pre_D)))])
        softmax = nn.Softmax(dim=1)
        for j in range(3):
            type_prototypes = type(Prototypes[j])
            if type_prototypes==dict:
                tensor_Prototype = torch.Tensor() #把字典变为tensor
                for i, val in Prototypes[j].items():
                    tensor_Prototype = torch.cat([tensor_Prototype,val.unsqueeze(0)],0)
            else:
                tensor_Prototype = Prototypes
                
            logits = torch.matmul(Pre_D[:,j,:], tensor_Prototype.T)
            Y = torch.argmax(softmax(logits),dim=1) 
            Y = Y.reshape((1,-1))
            Y_list = np.concatenate((Y_list,Y),axis=0)
        Y_list = Y_list[1:,:]
       # print(Y_list,'Y_list*************')
        Y_list_last = []
        for j in range(len(Y_list[0])):
            Y_list_last.append(Counter(Y_list[:,j]).most_common(1)[0][0])
        #print(Y_list_last,'^^^^^^^^^^^^')
        return torch.Tensor(Y_list[2])

    def predict(self, X, transformer, Prototypes, batch_size=256):
        """
        """
        X = tensor(X, dtype=torch.float32)
        if self.to_cuda:
            X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=len(X), shuffle=False, drop_last=False)

        # set model in eval mode
        self.model.eval()

        """Evaluate the given data loader on the model and return predictions"""
        result_latter = None
        result = None
        with torch.no_grad():
            for x in dl:
                Pre_D = self.model(x[0])  #Pre_D的维度为([256, 1, 54]) 
                if type(transformer)!=None:
                    Pre_D_latter, Prototypes_latter = transformer(Pre_D, Prototypes)
                else:
                    Pre_D_latter, Prototypes_latter = Pre_D, Prototypes
                '''transformer之后的结果'''
                Pre_D_latter = torch.squeeze(Pre_D_latter)
                y_hat_latter = self.predict_by_prototypes(Pre_D_latter, Prototypes_latter)  
                #y_hat_latter = self.predict_by_prototypes_limit(Pre_D_latter, Prototypes_latter)
                y_hat_latter = y_hat_latter.cpu().detach().numpy()
                result_latter = y_hat_latter if result_latter is None else np.concatenate((result_latter, y_hat_latter), axis=0)
                '''transformer之前的结果'''
                Pre_D = torch.squeeze(Pre_D)
                y_hat = self.predict_by_prototypes(Pre_D, Prototypes)  
                #y_hat = self.predict_by_prototypes_limit(Pre_D, Prototypes)
                y_hat = y_hat.cpu().detach().numpy()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result_latter, Prototypes_latter, result, Prototypes

    def get_shapelets(self):
        """
        不同类别模型的shapelets拼接在一起
        """
        return self.model.get_shapelets().clone().cpu().detach().numpy()

    def get_weights_linear_layer(self):
        """
        """
        return (self.model.linear.weight.data.clone().cpu().detach().numpy(),
                self.model.linear.bias.data.clone().cpu().detach().numpy())