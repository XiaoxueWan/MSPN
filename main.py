# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:37:54 2022

@author: Lenovo
"""
import pandas as pd
import torch
import time

from utils import get_weights_via_kmeans,plot_shapelets,eval_accuracy
from get_data.get_data import get_data_ucr,get_data_ACS,get_data_ucr_combine,get_data_TEP
from models.pre_trained_model import LearningShapeletsPretrain
from models.incremental_session import LearningShapelets
from models.meta_trainer import FakeTrainer
from matplotlib import pyplot
from torch import optim

class main():
    def __init__(self,
                     ucr_dataset_name,
                     ucr_dataset_base_folder,
                     pre_train_dataset_folder,
                     K,
                     Lmin,
                     temp_factor = 5, 
                     pre_train_temp_factor = 2,
                     learning_rate = 0.01,
                     learning_rate_meta = 0.01,
                     epoch = 2000,
                     epoch_meta_learning = 100,
                     batch_size = 234,
                     lw = 0.01,
                     is_incremental_learning = True,
                     number_class_each_task = 2,
                     variable_selection = 0,  
                     add_pre_trained = True,
                     dropout = 0.3,
                     c = 0.5
                     ):
        '''
            s_num: shapelet数量，K为shapelet数量占所有的比例
            s_length: shapelet的长度，Lmin为shapelet长度占所有的比例
            self.lr: 学习率
            self.lk: 蒸馏项损失
            self.temp_factor: 知识蒸馏的温度因子
            self.task_set: {
                            0 : {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test},
                            1 : {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test},
                            2 : {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test},
                            3 : {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
                            } 
                           不同key代表不同的任务
            number_splits: 表示将公共数据集切成多少份
        '''
        '''get data: Alu, TEP, UCR分割, UCR组合'''
        if ucr_dataset_name=='Alu':
            if add_pre_trained == True:
                #self.pre_train_X, self.pre_train_Y = get_data_ACS(pre_train_dataset_folder).main()
                self.task_set = get_data_ACS(ucr_dataset_base_folder,number_class_each_task).main()
            else:
                self.task_set = get_data_ACS(ucr_dataset_base_folder,number_class_each_task).main()
                self.knowledge = None
            self.pre_train_X = self.task_set[0]['X_train']
            self.pre_train_Y = self.task_set[0]['y_train']
        elif ucr_dataset_name=='TEP':
            if add_pre_trained == True:
                self.pre_train_X, self.pre_train_Y = get_data_TEP(pre_train_dataset_folder).main()
                self.task_set = get_data_TEP(ucr_dataset_base_folder).main()
            else:
                self.task_set = get_data_TEP(ucr_dataset_base_folder,number_class_each_task,variable_selection).main()
                self.pre_train_X = self.task_set[0]['X_train']
                self.pre_train_Y = self.task_set[0]['y_train']
        elif type(ucr_dataset_name)==list:
            self.task_set = get_data_ucr_combine(
                                                 ucr_dataset_name,
                                                 ucr_dataset_base_folder
                                                 ).main()
        else:
            self.task_set = get_data_ucr(
                                         ucr_dataset_name,
                                         ucr_dataset_base_folder,
                                         number_class_each_task
                                         ).main()
            self.pre_train_X = self.task_set[0]['X_train']
            self.pre_train_Y = self.task_set[0]['y_train']
         
        '''模型的一些参数'''
        self.add_pre_trained = add_pre_trained
        self.ucr_dataset_name = ucr_dataset_name
        self.lr = learning_rate
        self.lr_meta = learning_rate_meta
        self.w = lw
        self.epsilon = 1e-7
        
        self.temp_factor = temp_factor
        self.pre_train_temp_factor = pre_train_temp_factor
        self.is_incremental_learning = is_incremental_learning
        
        s_num = int(K*self.task_set[0]['X_train'].shape[0])   
        s_lenght = int(Lmin*self.task_set[0]['X_train'].shape[2])  
        #print(c*s_lenght, s_lenght,2*c*s_lenght, 3*c*s_lenght,'*********')
        self.shapelets_size_and_len = {int(c*s_lenght):s_num, int(2*c*s_lenght):s_num, int(3*c*s_lenght):s_num}   #定义shapelet的多个长度
        print(self.shapelets_size_and_len,'shapelets_size_and_len')
        
        self.epoch = epoch
        self.epoch_meta_learning = epoch_meta_learning
        self.batch_size = batch_size
        self.dropout = dropout
        t = time.localtime()
        
        self.record = {  
                         'time':str(t.tm_year) + '/'+ str(t.tm_mon) +'/'+ str(t.tm_mday)+'/'+ str(t.tm_hour)+':'+ str(t.tm_min), 
                         'ucr_dataset_name':ucr_dataset_name,
                         'ucr_dataset_path':ucr_dataset_base_folder, 
                         'K': K,
                         'Lim': Lmin, 
                         'is_incremental_learning':self.is_incremental_learning,
                         'temp_factor':temp_factor,
                         'pre_train_temp_factor':pre_train_temp_factor,
                         'Is_pre_training':add_pre_trained,
                         'epoch':self.epoch,
                         'epoch_meta':self.epoch_meta_learning,
                         'learning_rate': self.lr, 
                         'learning_rate_meta': self.lr_meta, 
                         'lw': self.w, 
                         'batch_size':self.batch_size, 
                         'Accuracy':0, 
                         'train_time':0, 
                         'test_time':0,
                         'dropout':self.dropout,
                         'c':c
                         }
    
    '''---------------------------------------模型初始化---------------------------------------------'''
    
    def initialize_shapelet_pretrain_optimizer(self, learning_shapelets): 
        '''
            @初始化预训练模型
        ''' 
        #初始化预训练模型的优化器
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            weights_block = get_weights_via_kmeans(self.task_set[0]['X_train'], shapelets_size, num_shapelets)
            learning_shapelets.set_shapelet_weights_of_block(i, weights_block)
            
        optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=self.lr, weight_decay=self.w, eps=self.epsilon)
        #optimizer = optim.SGD(learning_shapelets.model.parameters(), lr=self.lr, momentum=0.9)
        learning_shapelets.set_optimizer(optimizer)
        return learning_shapelets
    
    def initialize_pretrain_model(self,task): 
        '''
           @根据初始任务的数据对模型进行初始化
        '''
        n_ts,n_channels,len_ts = task['X_train'].shape
        num_classes = len(set(task['y_train']))
        dist_measure = 'euclidean'
      
        learning_shapelets_pretrained = LearningShapeletsPretrain(
                                                                   shapelets_size_and_len = self.shapelets_size_and_len,
                                                                   temp_factor = self.pre_train_temp_factor,
                                                                   in_channels = n_channels,
                                                                   num_classes = num_classes,
                                                                   to_cuda = False,
                                                                   verbose = 1,
                                                                   dist_measure = dist_measure,
                                                                   ucr_dataset_name = self.ucr_dataset_name
                                                                   )
        #对shapelet和优化器optimal进行初始化
        learning_shapelets_pretrained = self.initialize_shapelet_pretrain_optimizer(learning_shapelets_pretrained)
        return learning_shapelets_pretrained
        
    '''@初始化元模型'''
    def initialize_meta_training(self, meta_model, pretrain_model):
        '''
           @对模型的参数和优化方式进行初始化
        '''
        num_shapelets0 = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            meta_model.set_shapelet_weights_of_block(i, pretrain_model.get_shapelets()[num_shapelets0:num_shapelets0+num_shapelets,:,:shapelets_size])  #用预训练的shapelet作为元模型的初始化shapelets
            num_shapelets0 = num_shapelets0+num_shapelets
        optimizer = optim.SGD([{"params":meta_model.model.parameters()},
                                {"params":meta_model.slf_attn0.parameters()}, 
                                {"params":meta_model.slf_attn1.parameters()},
                                {"params":meta_model.slf_attn2.parameters()}],
                                lr=self.lr_meta, momentum=0.9)
        meta_model.set_optimizer(optimizer)
        return meta_model
    
     #
    def initialize_meta_training_model(self, pretrain_model):
        '''
            @初始化元模型
        '''
        meta_train = FakeTrainer(
                                   epoches_meta_train = self.epoch_meta_learning,
                                   shapelets_size_and_len = self.shapelets_size_and_len, 
                                   in_channels=1, 
                                   num_classes=2,
                                   dist_measure='euclidean',
                                   ucr_dataset_name='comman', 
                                   to_cuda=False,
                                   temp_factor=self.temp_factor,
                                   dropout0=self.dropout 
                                   )
        #初始化模型参数
        meta_train = self.initialize_meta_training(meta_train,pretrain_model)
        return meta_train
        
    
    def initialize_incremental_model(self):
        ''' 
            @初始化后面的增量阶段模型 
        '''
        self.learning_shapelets = LearningShapelets(
                                                   shapelets_size_and_len = self.shapelets_size_and_len,
                                                   in_channels = 1,
                                                   num_classes = 2,
                                                   to_cuda = False,
                                                   verbose = 1,
                                                   dist_measure = 'euclidean',
                                                   ucr_dataset_name = self.ucr_dataset_name
                                                   )
        
        return
      
    '''---------------------------------------对每个阶段的模型进行训练---------------------------------------------'''
    def pretrain(self): 
        ''' 
           对第0个阶段的数据集进行预训练
        '''
        learning_shapelets_pretrained = self.initialize_pretrain_model(
                                                                        self.task_set[0]
                                                                       )
        losses,Prototypes = learning_shapelets_pretrained.fit(
                                                                self.pre_train_X, 
                                                                self.pre_train_Y, 
                                                                epochs=self.epoch, 
                                                                batch_size=256, 
                                                                shuffle=False, 
                                                                drop_last=False
                                                                )
        return Prototypes, learning_shapelets_pretrained
        
    def train_single_task(self, task, num_task, Prototypes, learning_shapelets_last_task): 
        '''initialize incremetal session model'''
        learning_shapelets = LearningShapelets( shapelets_size_and_len = self.shapelets_size_and_len, 
                                                 in_channels = 1, 
                                                 dist_measure='euclidean',
                                                 ucr_dataset_name='comman', 
                                                 num_classes = 2)   #----------------------------------
        
        num_shapelets0 = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            learning_shapelets.set_shapelet_weights_of_block(i, learning_shapelets_last_task.get_shapelets()[num_shapelets0:num_shapelets0+num_shapelets,:,:shapelets_size])  #用预训练的shapelet作为元模型的初始化shapelets
            num_shapelets0 = num_shapelets0+num_shapelets
        
        '''task_change_to_train_test'''
        #print(task,'****************task')
        X_train = task['X_train']
        y_train = task['y_train']
        
        '''Training'''
        start = time.clock()
        Prototypes = learning_shapelets.fit(
                                            X_train, 
                                            y_train, 
                                            num_task,
                                            Prototypes,
                                            epochs=1, 
                                            batch_size=256, 
                                            shuffle=False, 
                                            drop_last=False
                                            )
        end = time.clock()
        self.record['train_time'] = end-start
        
        '''save model'''
        torch.save(learning_shapelets,'SIL.pth')
        return learning_shapelets,Prototypes
    
    
    '''---------------------------------------测试---------------------------------------------'''
    def test_single_task(self, learning_shapelets, X_test, y_test, transformer, Prototypes):
        '''测试单个任务'''
        start = time.clock()
        Accuracy, Prototypes = eval_accuracy(learning_shapelets, X_test, y_test, transformer, Prototypes)
        end = time.clock()
        test_time = end-start
        
        '''plot_shapelets'''
#        shapelets = learning_shapelets.get_shapelets()
#        shapelet_transform = learning_shapelets.transform(X_test)
#        record_data_plot = plot_shapelets(
#                                           X_test, 
#                                           shapelets, 
#                                           y_test, 
#                                           shapelet_transform, 
#                                           self.ucr_dataset_name,
#                                           self.shapelets_size_and_len
#                                           )
        return test_time, Accuracy, Prototypes
    
    def test_all_task(self, learning_shapelets, key, transformer, Prototypes):
        '''测试所有任务'''
        X_test = self.task_set[key]['X_test']
        y_test = self.task_set[key]['y_test']
       
        test_time, Accuracy, Prototypes = self.test_single_task(learning_shapelets, X_test, y_test, transformer, Prototypes)
      
        self.record['task'+str(key)+'Accuracy'] = Accuracy
            
        '''to_excel'''
        record = pd.read_excel('results/record_auto_11_13.xlsx')
        record = record.append(self.record, ignore_index = True)
        record = record.reindex(
                                 columns=['time',
                                          'ucr_dataset_name',
                                          'ucr_dataset_path',
                                          'K',
                                          'Lim', 
                                          'distillation_parameter',
                                          'epoch',
                                          'learning_rate', 
                                          'lw', 
                                          'batch_size', 
                                          'train_time', 
                                          'test_time',
                                          'Accuracy'
                                           ]+list(record.columns.drop(['time',
                                                                       'ucr_dataset_name',
                                                                       'ucr_dataset_path',
                                                                       'K',
                                                                       'Lim', 
                                                                       'distillation_parameter',
                                                                       'epoch',
                                                                       'learning_rate', 
                                                                       'lw', 
                                                                       'batch_size', 
                                                                       'train_time', 
                                                                       'test_time',
                                                                       'Accuracy'])))
        record.to_excel('results/record_auto_11_13.xlsx', index = False)
        return Prototypes 
    
    '''---------------------------------------main-------------------------------------------'''
    def incremental_learning_train(self):
        '''
        主函数：
                is_incremental_learning：判断是否是增量学习，是增量学习再判断是否是第一次任务，
                                        是第一次任务和不是增量学习is_first_task为True
        '''
        
        '''预训练''' 
        start = time.clock()
        pretrain_Prototypes, pretrain_model = self.pretrain()    #获取预训练模型
        '''元模型初始化'''
        meta_train_model = self.initialize_meta_training_model(pretrain_model)
       # print(self.task_set,'self.task_set')
        for num_task, value in self.task_set.items():
            if num_task == 0: 
                transformer, learning_shapelets_last_task, losses, Prototypes_0 = meta_train_model.meta_train(value, pretrain_Prototypes)
                self.test_all_task(pretrain_model, num_task, transformer, pretrain_Prototypes)
               
                #self.test_all_task(pretrain_model, num_task, None, pretrain_Prototypes)
                
                '''plot loss shapelet'''
                pyplot.plot(losses, color='black')
                pyplot.title("Loss over meta learning process")
                pyplot.show()
                Prototypes = pretrain_Prototypes
            else:
               # print(num_task,value,'*************Prototypes')
                start = time.clock()
                learning_shapelets, Prototypes = self.train_single_task(value, num_task, Prototypes, learning_shapelets_last_task)
                self.test_all_task(learning_shapelets, num_task, transformer, Prototypes)
                end = time.clock()
                print(end-start,'onlinetime**********')
               # learning_shapelets, Prototypes = self.train_single_task(value, num_task, pretrain_Prototypes, pretrain_model)
               # self.test_all_task(pretrain_model, num_task, None, Prototypes)
        end = time.clock()
        print(end-start,'**********')
        return
        
        
    