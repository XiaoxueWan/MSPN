# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:41:35 2022

@author: Lenovo
"""
import sys 
sys.path.append("..") 

import torch
import warnings
import numpy as np
from torch import tensor
from torch import nn
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .shapelet_network_pre_trained.shapelet_network import LearningShapeletsModel

class LearningShapeletsPretrain:
    """
  
    """
    def __init__(self, shapelets_size_and_len, temp_factor, in_channels=1, num_classes=2,
                 dist_measure='euclidean',ucr_dataset_name='comman', verbose=0, to_cuda=False, k=0, l1=0.0, l2=0.0):

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
        self.use_regularizer = False
        self.temp_factor = temp_factor

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
            
    def change_prototype_type_to_tensor(self, Prototypes):
        '''将各种类型的原型转换为tensor
           Input: 
               Prototypes: list [dict, dict, dict]
           Output: 
               Prototypes: list [tensor, tensor, tensor]
        '''
        type_prototypes = type(Prototypes[0]) #查看每个视图的Prototypes是什么类型,是字典还是tensor.
        Prototypes_list = []
        for i in range(3): #遍历每个视图
            if type_prototypes==dict:
                tensor_Prototype = torch.Tensor() #把字典变为tensor
                for i, val in Prototypes[i].items():
                    tensor_Prototype = torch.cat([tensor_Prototype,val.unsqueeze(0)],0)
            else:
                tensor_Prototype = Prototypes[i]
            Prototypes_list.append(tensor_Prototype)
        return Prototypes_list
            
    def prototype_class_loss(self, D, y, Prototypes0, temperature_parameter):
        '''
           定义一个原型的损失函数,与交叉熵损失不同在于没有分类器参数
           D:
                映射得到的特征  R:[num_samples,1,dim]
           Prototypes:
                 代表所有类别的原型集,假设总共是k个类别
                 --[视图1原型（tensor),视图2原型（tensor），视图3原型（tensor)] 
           similar_join: 
                 代表所有原型与每个样本对应特征集的距离. similar_join = exp(余弦相似度(D,p)/温度系数)
                 --NXk
           similar_sum: 
                 代表所有原型与每个样本对应特征集的距离. similar_join = 求和（exp(余弦相似度(D,p)/温度系数)）
                 --NX1
        '''
        Prototypes0 = self.change_prototype_type_to_tensor(Prototypes0)
        Prototypes = torch.Tensor()
        for Prototype in Prototypes0:
            Prototypes = torch.cat([Prototypes,Prototype.unsqueeze(1)],dim=1)
        Prototypes = Prototypes.expand(D.shape[0],Prototypes.shape[0],Prototypes.shape[1],Prototypes.shape[2])
        D = D.unsqueeze(1)
        similar = torch.cosine_similarity(D,Prototypes,dim=-1)  #similar:R[num_samples,num_protos]
       #用transformer 对特征维度和原型进行对齐
        similar = torch.mean(similar,dim=-1) #将视图的维度求均值
        D = D.to(torch.float32)
        cos = torch.exp(similar/temperature_parameter) 
        
        y = y.unsqueeze(1)
        similar = torch.gather(cos, dim=1, index=y)      #取计算损失函数的公式的分子，用y对余弦相似度做映射，因此可以取对应类别标签对应原型的余弦相似度
         #将每个原型对应的余弦相似度值进行拼接
    
        similar = torch.squeeze(similar)
        similar_sum = torch.sum(cos,dim=1)
        similar_div = torch.div(similar,similar_sum)
        return -torch.mean(torch.log(similar_div))
            
    def get_prototype(self, D, y):
        '''
           新来一个任务的数据，首选初始化计算当前任务对应的特征原型.
           Input
                D: [num_samples, num_shapelets] 
                    tensor 特征矩阵 
               y: [1,2,3] 
                   tensor X对应的标签 
               原型集 tensor Prototypes:tensor,[[],[],[]] R:原型数*num_shapelets
           Output
              Prototypes: [num_Prototypes, num_shapelets] tensor
        '''
        y = y.unsqueeze(1)
        y = y.float()
        D_y = torch.cat((D,y),1) #将D拼上y 
        y = torch.squeeze(y)
        y = y.numpy()
        Prototypes_current_task = dict.fromkeys(list(set(y)))  #新建一个空的字典，以y的值作为键
        for i in list(set(y)):
           mask = torch.where(D_y[:,-1]==i)
           D_y = D_y.float()
           D_y_class = torch.mean(D_y[mask][:,:-1],0)
           if Prototypes_current_task[i] == None:
               Prototypes_current_task[i] = D_y_class
     
        return Prototypes_current_task

    def update(self, x, y):
        """
           变成增量的模型更新方式
        """
        D0 = self.model(x)
        Prototypes_multi_view = []  #用于存放多个视图的原型
        for i in range(3): #3是视图数
            #print(D0.shape,'D0.shape((((((((((')
            D = D0[:,i,:]
            Prototypes_current_task1 = self.get_prototype(D,y)
            Prototypes_multi_view.append(Prototypes_current_task1)
        loss = self.prototype_class_loss(D0, y, Prototypes_multi_view, self.temp_factor)
       
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), Prototypes_multi_view

    def fit(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        """
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

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
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

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
                current_loss_ce, Prototypes = self.update(x, y)
                losses_ce.append(current_loss_ce)
        return losses_ce, Prototypes if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
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
        #print(Pre_D.shape,'Pre_D.shape***********')
        
        Y_list = []
        for i in range(3): #遍历每个原型
            distance_proto_D = torch.tensor([])
            for classes,Prototype in  Prototypes[i].items():
                cos = torch.cosine_similarity(Pre_D[:,i,:],torch.tensor(Prototype),dim=-1)
                cos = cos.unsqueeze(1)
                distance_proto_D = torch.cat((distance_proto_D,cos),dim=1)
            Y_list.append(torch.argmax(distance_proto_D,dim=1))
        #print(Y_list,'Y_list**************')
        return Counter(Y_list).most_common(1)[0][0]
    
    def predict_by_prototypes_limit(self, Pre_D, Prototypes):
        softmax = nn.Softmax(dim=1)
        type_prototypes = type(Prototypes)
        if type_prototypes==dict:
            tensor_Prototype = torch.Tensor() #把字典变为tensor
            for i, val in Prototypes.items():
                tensor_Prototype = torch.cat([tensor_Prototype,val.unsqueeze(0)],0)
        else:
            tensor_Prototype = Prototypes
            
        logits = torch.matmul(Pre_D, tensor_Prototype.T)
        return torch.argmax(softmax(logits),dim=1), Prototypes

    def predict(self, X, transformer=None, Prototypes=None, batch_size=256):
        """
        """
        X = tensor(X, dtype=torch.float32)
        if self.to_cuda:
            X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # set model in eval mode
        self.model.eval()

        """Evaluate the given data loader on the model and return predictions"""
        result = None
        result0 = None
        with torch.no_grad():
            for x in dl:
                Pre_D = self.model(x[0])
                Pre_D0, Prototypes0 = Pre_D, Prototypes
                
                y_hat= self.predict_by_prototypes(Pre_D, Prototypes)  
                y_hat = y_hat.cpu().detach().numpy()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
                
                y_hat0 = self.predict_by_prototypes(Pre_D0, Prototypes0)  
                y_hat0 = y_hat0.cpu().detach().numpy()
                result0 = y_hat0 if result0 is None else np.concatenate((result0, y_hat0), axis=0)
        return result, Prototypes, result0, Prototypes0

    def get_shapelets(self):
        """
        """
        return self.model.get_shapelets().clone().cpu().detach().numpy()

    def get_weights_linear_layer(self):
        """
        """
        return (self.model.linear.weight.data.clone().cpu().detach().numpy(),
                self.model.linear.bias.data.clone().cpu().detach().numpy())