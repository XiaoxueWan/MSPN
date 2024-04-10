# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:27:05 2023

@author: WXX
"""
import torch
import random
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from statistics import mean
from torch import tensor
import torch.nn.functional as F
from .shapelet_network.shapelet_network import LearningShapeletsModel
from .transformer.net import MultiHeadAttention0, MultiHeadAttention1, MultiHeadAttention2

class FakeTrainer:
    def __init__(self, 
                     epoches_meta_train,
                     shapelets_size_and_len, 
                     in_channels=1, 
                     num_classes=2,
                     dist_measure='euclidean',
                     ucr_dataset_name='comman', 
                     to_cuda=False,
                     temp_factor=2,
                     dropout0=0.3
                     ):
        
        self.temp_factor = temp_factor
        self.epoches_meta_train = epoches_meta_train
        
        self.model = LearningShapeletsModel(
                                            shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=1, 
                                            num_classes=num_classes, 
                                            dist_measure=dist_measure,
                                            ucr_dataset_name=ucr_dataset_name,
                                            to_cuda=to_cuda
                                            ) 
        
        '''多视图分别对应的模型'''
        self.slf_attn0 = MultiHeadAttention0(
                                               1, 
                                               list(shapelets_size_and_len.values())[0], #表示输入数据的特征维度
                                               list(shapelets_size_and_len.values())[0], 
                                               list(shapelets_size_and_len.values())[0], 
                                               dropout = dropout0
                                               )
        
        self.slf_attn1 = MultiHeadAttention1(
                                               1, 
                                               list(shapelets_size_and_len.values())[1], #表示输入数据的特征维度
                                               list(shapelets_size_and_len.values())[1], 
                                               list(shapelets_size_and_len.values())[1], 
                                               dropout = dropout0
                                               )
            
        self.slf_attn2 = MultiHeadAttention2(
                                               1, 
                                               list(shapelets_size_and_len.values())[2], #表示输入数据的特征维度
                                               list(shapelets_size_and_len.values())[2], 
                                               list(shapelets_size_and_len.values())[2], 
                                               dropout = dropout0
                                               )
        
        
    def set_optimizer(self, optimizer):
        """
        """
        self.optimizer = optimizer
        return
    
    def set_shapelet_weights_of_block(self, i, weights):
        """
        """
        self.model.set_shapelet_weights_of_block(i, weights)
        return
    
    def get_shapelets(self):
        """
        """
        return self.model.get_shapelets().clone().cpu().detach().numpy()
        
    def generate_support_query(self, base_dataset):
        '''
           随机生成 支持集support 和 查询集query, 根据论文《Few-Shot Class-Incremental Learning by Sampling Multi-Phase Tasks》中的Algorithm 1的伪代码
           input:
                base_dataset: {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
           output:
               support_set:{
                            0:{'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test},  #class:0
                            1:{'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}   #class:1
                            }
               query_set:{
                          0:{'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}, #class:0
                          1:{'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}  #class:0,1
                          }
               base_class_number_dict: 3, 表示基础任务的任务id
               
        '''
        train = np.concatenate((base_dataset['X_train'].reshape(base_dataset['X_train'].shape[0],base_dataset['X_train'].shape[-1]),
                                base_dataset['y_train'].reshape(base_dataset['y_train'].shape[0],1)), axis=1)
        test = np.concatenate((base_dataset['X_test'].reshape(base_dataset['X_test'].shape[0],base_dataset['X_test'].shape[-1])
                               ,base_dataset['y_test'].reshape(base_dataset['y_test'].shape[0],1)), axis=1)
        
        class_list = list(set(np.reshape(base_dataset['y_test'],(len(base_dataset['y_test'])))))
        #print(class_list,'&&&&&&&&&class_list')
        class_list = [int(x) for x in class_list]
        base_class_number = random.choice(class_list)
        
        keys = [a for a in class_list if a!=base_class_number] 
        support_x_dict =  dict.fromkeys(keys,{})    #新建空的支持集和查询集，任务数量由除了任务0以外的其他任务组成
        support_y_dict =  dict.fromkeys(keys,{})
        query_x_dict =  dict.fromkeys(keys,{})
        query_y_dict =  dict.fromkeys(keys,{})
        
        for i in class_list:
            if i!=base_class_number:
                mask = np.isin(train[:, -1],[i])
                X_train = train[mask][:100, :-1]
                y_train = train[mask][:100, -1]
                support_x_dict[i] = X_train.reshape(X_train.shape[0],1,X_train.shape[-1])
                support_y_dict[i] = y_train
                
                if base_class_number in list(range(i)):
                    mask = np.isin(test[:, -1],[i]+list(range(i)))
                    X_test = test[mask][:50, :-1]
                    y_test = test[mask][:50, -1]
                    query_x_dict[i] = X_test.reshape(X_test.shape[0],1,X_test.shape[-1])
                    query_y_dict[i] = y_test
                else:
                    mask = np.isin(test[:, -1],[i]+list(range(i))+[base_class_number])
                    X_test = test[mask][:50, :-1]
                    y_test = test[mask][:50, -1]
                    query_x_dict[i] = X_test.reshape(X_test.shape[0],1,X_test.shape[-1])
                    query_y_dict[i] = y_test
        return support_x_dict,support_y_dict,query_x_dict,query_y_dict,base_class_number
    
    def get_prototype(self, D, y, pretrain_prototypes_dict, fake_task_num, base_class_number, Prototypes):
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
        num_proto = list(set(y))  #新建一个空的字典，以y的值作为键
        for i in num_proto:
           mask = torch.where(D_y[:,-1]==i)
           D_y = D_y.float()
           D_y_class = torch.mean(D_y[mask][:,:-1],0)
           D_y_class = D_y_class.unsqueeze(0)
           if fake_task_num==0: #假如是假任务的第1个增量任务，假任务是从第一个开始
               prototypes = torch.cat([pretrain_prototypes_dict[base_class_number].unsqueeze(0),D_y_class],dim=0)
           else:
               prototypes = torch.cat([Prototypes,D_y_class],dim=0)
        return prototypes,num_proto
    
    def prototype_class_loss(self, D, y, Prototypes0, temperature_parameter, num_class_list):
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
        Prototypes = torch.Tensor()
        for Prototype in Prototypes0:
            Prototypes = torch.cat([Prototypes,Prototype],dim=1)
        
        Prototypes = Prototypes.expand(D.shape[0],Prototypes.shape[0],Prototypes.shape[1],Prototypes.shape[2])
        D = D.unsqueeze(1)
        
        similar = torch.cosine_similarity(D,Prototypes,dim=-1)  #similar:R[num_samples,num_protos]
       #用transformer 对特征维度和原型进行对齐
        similar = torch.mean(similar,dim=-1) #将视图的维度求均值
        
        D = D.to(torch.float32)
        cos = torch.exp(similar/temperature_parameter) 
        y_index = []
        for value in y:
           y_index.append(num_class_list.index(value))
        y_index = torch.tensor(y_index)
        y_index = y_index.unsqueeze(1)
        similar = torch.gather(cos, dim=1, index=y_index)      #取计算损失函数的公式的分子，用y对余弦相似度做映射，因此可以取对应类别标签对应原型的余弦相似度
         #将每个原型对应的余弦相似度值进行拼接
    
        similar = torch.squeeze(similar)
        similar_sum = torch.sum(cos,dim=1)
        similar_div = torch.div(similar,similar_sum)
        return -torch.mean(torch.log(similar_div))
    
    def loss_Limit(self, D0, y, Prototypes, num_class_list):
        '''
           limit方法里面用到的损失函数
           D:
                映射得到的特征  R:[num_samples,dim_views,dim]
           Prototypes:
                 代表所有类别的原型集,假设总共是k个类别
                 --[视图1原型（tensor),视图2原型（tensor），视图3原型（tensor)] 
        '''
        softmax = nn.Softmax(dim=1)
        #print(D0.shape,Prototypes[0].shape,'&&&&&&&&&&D.shape')
        Y_loss = torch.Tensor([0])
        
        y_index = []
        for value in y:
           y_index.append(num_class_list.index(value))
        y_index = torch.tensor(y_index)
        
        for i in range(3):
            D = D0[:,i,:]
            Prototype = Prototypes[i].T
           # print(D.shape, Prototypes[i].T.shape)
            logits = torch.matmul(D, torch.squeeze(Prototype))
            #print(logits.shape,softmax(logits.detach()).shape,'*******')
            loss = F.cross_entropy(softmax(logits), y_index)
           # print(loss,Y_loss,'&&&&&&&&&&&')
            loss = loss.unsqueeze(0)
            Y_loss = torch.cat((Y_loss,loss),dim=0)
    
        loss = torch.mean(Y_loss)
        return loss
    
    def change_prototype_type_to_tensor(self, Prototypes):
        '''将各种类型的原型转换为tensor'''
        type_prototypes = type(Prototypes) #查看每个视图的Prototypes是什么类型,是字典还是tensor.
        if type_prototypes==dict:
            tensor_Prototype = torch.Tensor() #把字典变为tensor
            for i, val in Prototypes.items():
                tensor_Prototype = torch.cat([tensor_Prototype,val.unsqueeze(0)],0)
            prototypes_num = len(Prototypes.keys())
        else:
            tensor_Prototype = Prototypes
            prototypes_num = tensor_Prototype.shape[0]
        return tensor_Prototype, type_prototypes, prototypes_num
    
    def change_prototype_tensor_to_type(self, tensor_Prototype, type_prototypes):
        '''将各种tensor的原型转换为字典或者其他类型'''
        if type_prototypes==dict:
            Prototypes = dict.fromkeys(list(range(tensor_Prototype.shape[0])))
            for i, val in Prototypes.items():#'再把tensor_Prototype 变为字典Prototype'
                Prototypes[i] = tensor_Prototype[i]
        else:
            Prototypes = tensor_Prototype.unsqueeze(1)
        return Prototypes
    
    def each_view_combine(self, tensor_Prototype, D):
        '''获取每个视图的结合数据：[D, prototype]
            Prototypes： [num_Prototypes, num_shapelets]
            D:[num_samples,1,num_shapelets]
        '''
        tensor_Prototype = F.normalize(tensor_Prototype,dim=1)  #将数据标准化到0-1之间
        D = F.normalize(D,dim=-1)
        
        tensor_Prototype = tensor_Prototype.unsqueeze(0).expand(D.shape[0],tensor_Prototype.shape[0],tensor_Prototype.shape[1]).contiguous()
        combined = torch.cat([tensor_Prototype, D],1)
        return combined
    
    def change_D_Proto_multi_view(self, D_view0, Prototypes_multi_view):
        '''
            利用transformer中的自注意力机制，变换D和原型Prototypes,从而对齐
            Input:
                D_view: [num_samples,num_view, num_shapelets]  shapelet 映射特征
                Prototypes_multi_view: [视图1原型（tensor),视图2原型（tensor），视图3原型（tensor)] 
            Output:
                D_views: [num_samples,num_view, num_shapelets]
                Prototypes_multi_view:  [num_prototypes, num_view, num_shapelets] 
        '''
        Prototypes_views = []
        D_views = torch.Tensor()
        for i in range(3): #遍历每个视图
            Prototypes_view, type_Prototypes_view, prototypes_num = self.change_prototype_type_to_tensor(Prototypes_multi_view[i])
            combined_view = self.each_view_combine(Prototypes_view, D_view0[:,i,:].unsqueeze(1))
            if i==0:
                combined_view_latter, combined_view0_cross_view = self.slf_attn0(combined_view, combined_view, combined_view)  #用transformer进行变换
            elif i==1:
                combined_view_latter, combined_view1_cross_view = self.slf_attn1(combined_view, combined_view, combined_view, combined_view0_cross_view, combined_view0_cross_view)  #用transformer进行变换
            else:
                combined_view_latter = self.slf_attn2(combined_view, combined_view, combined_view, combined_view1_cross_view, combined_view1_cross_view)  #用transformer进行变换
            tensor_Prototype_view, D_view = combined_view_latter.split(prototypes_num, 1)
           # print(i,tensor_Prototype_view.shape,'*********tensor_Prototype_view')
            tensor_Prototype_view = tensor_Prototype_view.narrow(0,0,1)
            tensor_Prototype_view = torch.squeeze(tensor_Prototype_view)
            
            Prototypes_view = self.change_prototype_tensor_to_type(tensor_Prototype_view, type_Prototypes_view)
            
            Prototypes_views.append(Prototypes_view)  #把不同视图的数据整合在一起
            D_views = torch.concat([D_views,D_view],dim=1)
        return D_views, Prototypes_views 
    
    def meta_train(self, base_dataset, pretrain_prototypes_list):
        '''
          元训练
          
          参数：
          ------------------------------------------------------------------------------------
          base_dataset: 基本数据集
          
          pretrain_prototypes_list: 
                                 [视图1原型（tensor),视图2原型（tensor），视图3原型（tensor)]
            
         返回值：
        -------------------------------------------------------------------------------------
        change_D_Proto_multi_view: 变换函数
        
        loss_plot：所有损失
                   []
                   
        Prototypes_multi_views： 多视图原型
                               [视图1， 视图2， 视图3]
        '''
        progress_bar = tqdm(range(self.epoches_meta_train))
        loss_plot = [] #记录损失函数
        for i in progress_bar:
            torch.cuda.empty_cache()
            torch.autograd.set_detect_anomaly(True)
            
            support_x_dict, support_y_dict, query_x_dict, query_y_dict, base_class_number = self.generate_support_query(base_dataset) #生成随机的支持集和查询集
            
            loss, fake_task_num, Prototypes_multi_views = 0,0,[0,0,0]
            proto_index = [int(base_class_number)]
            
            for key,value in support_x_dict.items():#遍历查询集和支持集中的每个样本，即遍历每个增量阶段
                support_x, support_y, query_x, query_y = support_x_dict[key], support_y_dict[key], query_x_dict[key], query_y_dict[key]
                #print(support_x.shape,support_y.shape,query_y.shape,base_class_number,'*******D_view')
                #print(key,'^^^^^^keysupport_y,support_x.shape')
                if not isinstance(support_x, torch.Tensor): #将支持集和查询集进行实例化
                    support_x = tensor(support_x, dtype=torch.float).contiguous()
                if not isinstance(query_x, torch.Tensor):
                    query_x = tensor(query_x, dtype=torch.float).contiguous()
                if not isinstance(support_y, torch.Tensor):
                    support_y = tensor(support_y, dtype=torch.long).contiguous()
                if not isinstance(query_y, torch.Tensor):
                    query_y = tensor(query_y, dtype=torch.long).contiguous()
                    
                D_support = self.model(support_x)
                D_support = D_support.detach() 
        
                D_query = self.model(query_x)
                #print(D_query.shape,'*******D_view')
                '''得到多视图原型'''
                Prototypes_multi_view = []  #用于存放多个视图的原型
                for i in range(3): #3是视图数
                    D_view = D_support[:,i,:]
                    
                    Prototypes1, num_proto = self.get_prototype(D_view, support_y, pretrain_prototypes_list[i], fake_task_num, base_class_number, Prototypes_multi_views[i])
                    Prototypes_multi_view.append(Prototypes1)
                
                Prototypes_multi_views = Prototypes_multi_view
                proto_index = proto_index + num_proto
                
                '''将多视图原型和特征矩阵进行变换'''
                D, Prototypes0 = self.change_D_Proto_multi_view(D_query, Prototypes_multi_views)  #用transformer 对特征维度和原型进行对齐
                loss = loss + self.prototype_class_loss(D, query_y, Prototypes0, self.temp_factor, proto_index)
                #loss = loss + self.loss_Limit(D, query_y, Prototypes0, proto_index)
                fake_task_num +=1
            loss_plot.append(loss)  #用于绘制损失图
            progress_bar.set_description(f"Loss: {loss}")
            
            with torch.autograd.detect_anomaly():
                loss.backward(retain_graph=True)
                
            '''迭代优化'''
            self.optimizer.step()
            self.optimizer.zero_grad()
        return self.change_D_Proto_multi_view, self.model, loss_plot, Prototypes_multi_views #返回训练好的元校正机制