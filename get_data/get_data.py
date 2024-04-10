# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:05:30 2022

@author: Lenovo
"""


from torch.utils.data import DataLoader, TensorDataset
from utils import normalize_data,preprocess0,normalize_y,load_UCR,normalize_label,Znormalize_dim2,normalize_data_0_1

import os
import torch
import pandas as pd
import numpy as np


class get_data_ucr_combine():
    def __init__(self,ucr_dataset_name,ucr_dataset_base_folder):
        ''' 
           将两个不同领域的数据集进行训练。每个数据集代表一个task
           ucr_dataset_name:[数据集1，数据集2]
           ucr_dataset_base_folder：所有数据集路径
        '''
        self.ucr_dataset_name = ucr_dataset_name
        self.ucr_dataset_base_folder = ucr_dataset_base_folder
        
    def load_single_task_dataset(self,task_id):
        #if self.ucr_dataset_base_floder!='zhonglv':
        train_raw_arr,test_raw_arr = load_UCR(self.ucr_dataset_base_folder, self.ucr_dataset_name[task_id])
        
        # training dat
        train_data = train_raw_arr[:, 1:]
        train_labels = train_raw_arr[:, 0] - 1
        
        # test_data 
        test_data = test_raw_arr[:, 1:]
        test_labels = test_raw_arr[:, 0] - 1
        #else:
        #print(train_data.shape,train_labels.shape,test_data.shape,test_labels.shape,'*******************')
        return train_data, train_labels, test_data, test_labels
        
    def main(self):
        Task = {i:[] for i in range(len(self.ucr_dataset_name))}
        
        for task_id,task in Task.items(): 
            X_train, y_train, X_test, y_test = self.load_single_task_dataset(task_id)
            
            y_train, y_test = normalize_y(y_train,y_test)
            X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
            
            Task[task_id] = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}
        return Task
    
class get_data_TEP():
    def __init__(self,ucr_dataset_base_folder,number_class_each_task,variable_selection):
        '''
         __init__:
            函数初始化
        Args:
            ucr_dataset_name:[dataname1,dataname2]
            ucr_dataset_path:文件路径
            variable_selection:int 选择变量
            variable_dict:{} 变量对应的变量名
        '''
        self.ucr_dataset_base_folder = ucr_dataset_base_folder
        self.number_class_each_task = number_class_each_task
        self.variable_selection = variable_selection
        self.variable_dict = {3:'A_and_C_feed',16:'Stripper underflow',17:'Stripper_temperature', 21:'Separator_cooling_water_outlet_temperature'}
        
    def data_label(self,files,file_choose):
        """
        data_label:
            将所有子文件的路径记录下来，并选择‘simout’或者'xmv'
        Args:
            files:所有文件路径
            file_choose:从文件里面选择‘simout’或者'xmv'
        """
        label_y=[]
        total_path=[]
        for filenames,dirnames,files in os.walk(files):
            if dirnames==[] and file_choose in filenames:
                for name in files:
                    #print(filenames,'&&&&&&&&&&&&name**************')
                    if '~' not in name and self.variable_dict[self.variable_selection] in name: 
                        total_path.append(filenames+'/'+name)
                        if any(char in filenames for char in ['10','11','12','13','14','15']):
                            label_y.append(int(filenames[-2:]))
                        else:
                            label_y.append(int(filenames[-1:]))
        return total_path,label_y
    
    def load_single_task_dataset(self,path,task_id):
        total_path_,y_total= self.data_label(path,'simout')
       # print(total_path_,'&&&&&&&&&&total_path')
        total = np.array([np.array(preprocess0(pd.read_excel(total_path)).T) for total_path in total_path_])
         #将顺序打乱
        index = [i for i in range(len(total))]
        np.random.shuffle(index) 
        total = total[index]
        #total, scalar = normalize_data(total)
        
        train_num = int(total.shape[0]*0.8)
        if task_id == 0:
            X_train = total[:train_num,:,:]   
            X_test = total[train_num:,:,:]
        else:
            X_train = total[:5,:,:]   #假如不是第一个阶段,那么训练数据应该是小样本
            X_test = total[5:10,:,:]
        #数据标签 
        y_total = np.array(y_total)
      #  y_total = normalize_label(y_total)
        y_total = y_total[index]
        
        if task_id == 0:
            y_train = y_total[:train_num]   
            y_test = y_total[train_num:]
        else:
            y_train = y_total[:5]   #假如不是第一个阶段,那么训练数据应该是小样本
            y_test = y_total[5:10]
        #y_train = y_total[:train_num]
        return X_train,y_train,X_test,y_test
    
    def main(self):
        task_classes_dict = self.number_class_each_task
        Task = {i:[] for i in range(len(task_classes_dict))}
        X_test, y_test = None,None
        if type(self.ucr_dataset_base_folder)==list:
            for task_id,task in task_classes_dict.items():  
                X_train, y_train = None,None
                for flod_id in task:
                    X_train0, y_train0, X_test0, y_test0 = self.load_single_task_dataset(self.ucr_dataset_base_folder[flod_id],task_id)
                    X_train = X_train0 if X_train is None else np.concatenate((X_train0, X_train), axis=0)
                    y_train = y_train0 if y_train is None else np.concatenate((y_train0, y_train), axis=0)
                    X_test = X_test0 if X_test is None else np.concatenate((X_test0, X_test), axis=0)
                    y_test = y_test0 if y_test is None else np.concatenate((y_test0, y_test), axis=0)
                print(X_train.shape,len(y_train),X_test.shape,len(y_test),'y_trainX_test***********')
                Task[task_id] = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}    
            return Task
        else:
            X_train, y_train, X_test, y_test = self.load_single_task_dataset(self.ucr_dataset_base_folder)
            return X_train, y_train

class get_data_ucr():
    def __init__(self, ucr_dataset_name,ucr_dataset_base_folder,number_class_each_task):
        '''
           将一个公共数据集切分成多个task
           ucr_dataset_name:数据集名字
           ucr_dataset_base_folder：数据集路径
           number_splits：数据切分数量
        '''
        self.ucr_dataset_name = ucr_dataset_name
        self.ucr_dataset_base_folder = ucr_dataset_base_folder
        
        self.number_class_each_task = number_class_each_task
    
    def load_dataset(self):
        
        #print('train_file_path',train_file_path)
        train_raw_arr,test_raw_arr = load_UCR(self.ucr_dataset_base_folder, self.ucr_dataset_name)
        #print(train_raw_arr.shape,test_raw_arr.shape,'train_raw_arr.shape')
        # training data
        train_labels = train_raw_arr[:, 0] - 1
        #print(train_labels.shape,'train_labels.shape')
        #else:
        #print(train_data.shape,train_labels.shape,test_data.shape,test_labels.shape,'*******************')
        self.num_classes = len(list(set(train_labels)))
        return train_raw_arr,test_raw_arr
    
    def build_task_class_set(self):
        '''
        构建每个任务学习的类别集
           self.number_class_each_task：代表每个任务识别多少个类别数
           输出 例如：{task0:[class1],task1:[class2],task2:[class3]}
           假如类别数为4,每个任务类别数为2：
              输出：{0: [1, 2], 1: [3, 4]}
           假如类别数为4,每个任务类别数为1：
              输出：{0: [1], 1: [2], 2: [3], 3: [4]}
           假如类别数为5,每个任务类别数为2
              输出：{0: [1, 2], 1: [3, 4], 2: [5]}
        '''
        task_classes_test_dict = {}
        if type(self.number_class_each_task) == dict:  #如果是人为指定的，就直接用
            task_classes_dict = self.number_class_each_task
            for key,value in self.number_class_each_task.items():
                if key==0:
                    task_classes_test_dict[key]=value
                else:
                    task_classes_test_dict[key]= value + task_classes_test_dict[key-1]
        else:
            num_task = int(self.num_classes/self.number_class_each_task)
            if self.num_classes%self.number_class_each_task == 0:  #如果类别数能够整除每个类别识别的类别个数,基任务分类的类别数为每个任务识别的任务数，否则为余数
                num_task_last = self.number_class_each_task
                task_classes_dict = {i:[] for i in range(num_task)}
            else:
                num_task_last = self.num_classes%self.number_class_each_task
                task_classes_dict = {i:[] for i in range(0,num_task+1,1)}
        
            for task_id,classes in task_classes_dict.items():#task_classes_dict[task_id-1][-1]代表上一次任务的类别列表的最后一个数值
                if task_id==len(task_classes_dict)-1 and task_id>0:
                    task_classes_dict[task_id] = list(range(task_classes_dict[task_id-1][-1]+1,num_task_last+task_classes_dict[task_id-1][-1]+1,1))
                elif task_id==0:
                    task_classes_dict[task_id] = list(range(1,self.number_class_each_task+1,1))
                else:
                    task_classes_dict[task_id] = list(range(task_classes_dict[task_id-1][-1]+1,task_classes_dict[task_id-1][-1]+1+self.number_class_each_task,1))
      #  print(task_classes_test_dict,'task_classes_test_dict**********')
        return task_classes_dict, task_classes_test_dict
                
    def main(self):
        '''
           对公共数据集进行处理，Task用于存放不同任务的公共数据集，用分割数量对公共数据集进行切分
        '''
        train_raw_arr,test_raw_arr = self.load_dataset()
        #每个任务的类别标签
        task_classes_dict, task_classes_test_dict = self.build_task_class_set()
        #print(task_classes_dict,'***********task_classes_dict********')
        Task = {i:[] for i in range(len(task_classes_dict))}
        #print(X_train.shape,type(X_train),y_train.shape,type(y_train),'X_train.shape,type(X_train),y_train.shape,type(y_train)')
        for task_id,task in Task.items():
            if self.number_class_each_task==0 and type(self.number_class_each_task)==int:
                #print(task_classes_dict[task_id][-1],'task_classes_dict[task_id][-1]')
                mask = np.logical_not(train_raw_arr[:, 0] != task_classes_dict[task_id][-1])
                X_train = train_raw_arr[mask][:, 1:]
                y_train = train_raw_arr[mask][:, 0]-1
                
                mask_test_y = np.logical_not(test_raw_arr[:, 0] != task_classes_dict[task_id][-1])
                X_test = test_raw_arr[mask][:, 1:]
                y_test = test_raw_arr[mask_test_y][:, 0]-1
            else: 
                mask = np.isin(train_raw_arr[:, 0],task_classes_dict[task_id])
                if task_id !=0:
                    X_train = train_raw_arr[mask][:10, 1:]
                    y_train = train_raw_arr[mask][:10, 0]-1
                    
                    mask_test_x = np.isin(test_raw_arr[:, 0],task_classes_test_dict[task_id])
                    mask_test_y = np.isin(test_raw_arr[:, 0],task_classes_test_dict[task_id])
                    X_test = test_raw_arr[mask_test_x][:10, 1:]
                    y_test = test_raw_arr[mask_test_y][:10, 0]-1
                else:
                    X_train = train_raw_arr[mask][:, 1:]
                    y_train = train_raw_arr[mask][:, 0]-1
                    
                    mask_test_x = np.isin(test_raw_arr[:, 0],task_classes_test_dict[task_id])
                    mask_test_y = np.isin(test_raw_arr[:, 0],task_classes_test_dict[task_id])
                    X_test = test_raw_arr[mask_test_x][:, 1:]
                    y_test = test_raw_arr[mask_test_y][:, 0]-1
            #y_train, y_test = normalize_y(y_train,y_test)
           # X_train = normalize_data_0_1(X_train)
           # X_test = normalize_data_0_1(X_test)
            
            X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
            Task[task_id] = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}
        #当标签不是0,1,2这种，而是1,2这种时
        return Task
    
class get_data_ACS():
    def __init__(self,ucr_dataset_base_folder,number_class_each_task):
        """
        Args:
            slef.path_total:输入数据路径
            self.path_adj:邻接矩阵路径
        """
        self.ucr_dataset_base_folder = ucr_dataset_base_folder
        self.number_class_each_task = number_class_each_task
        
    def csv_to_excel(self,path_total):
        """
        csv_to_excel:
            将数据中csv的数据转换为excel
        Args:
            path_total:包含所有文件路径的列表
        """
        for i in path_total:
            if i[-4:]=='.csv':
                csv=pd.read_csv(i,encoding='utf-8',header=None,engine='python')
                csv.to_excel(i.replace('.csv','.xlsx'),encoding='utf-8')
                os.remove(i)
        return 
    
    def del_excess_columns_indexs(self,path_total):
        """
        del_excess_columns_indexs:
            如果数据包含多余的行和列就将多余的行列删除，并写入新的excel中
        Args:
            path_total:包含所有文件路径的列表
        """
        for i in path_total:
            if i[-4:]=='.csv':
                csv=pd.read_csv(i,header=None,engine='python')
            else:
                csv=pd.read_excel(i,header=None)
            if pd.isnull(csv.iloc[0,0]):
                print(i)
                csv=csv.drop(csv.index[[0]])
                csv=csv.drop(csv.columns[[0]],axis=1)
                csv.to_excel(i,header=None,index=False)
        return
    
    def change_data_total_all(self,path_total):
        """
        change_data：
             对数据进行处理,将csv数据转为excel数据,删除出现多余第一行和第一列的数据
        Args:
            total_path_：处理完之后的数据的路径列表
            y_total：所有的标签列表
        """
        total_path_,y_total=self.data_label(path_total)
        self.csv_to_excel(total_path_)
        total_path_,y_train=self.data_label(path_total)
        self.del_excess_columns_indexs(total_path_)
        return total_path_,y_total
    
    def data_label(self,files):
        """
        data_label:
            获取数据和标签
        Args:
            total_path:用于记录所有文件的路径
            label_y:用于记录每个文件的标签
        """
        label_y=[]
        total_path=[]
        for filenames,dirnames,files in os.walk(files):
            for name in files:
                total_path+=[filenames+'/'+name]
                label_y.append(int(filenames[-1]))
        return total_path,label_y
    

    def load_single_task_dataset(self,path,task_id):
        total_path_,y_total= self.change_data_total_all(path)
       # print(total_path_,'&&&&&&&&&&total_path')
        total = np.array([np.array(preprocess0(pd.read_excel(total_path)[['potVolt']]).T) for total_path in total_path_])
         #将顺序打乱
        index = [i for i in range(len(total))]
        np.random.shuffle(index) 
        total = total[index]
        #total, scalar = normalize_data(total)
        
        train_num = int(total.shape[0]*0.8)
        if task_id == 0:
            X_train = total[:train_num,:,:]   
            X_test = total[train_num:,:,:]
        else:
            X_train = total[:5,:,:]   #假如不是第一个阶段,那么训练数据应该是小样本
            X_test = total[5:10,:,:]
        #数据标签 
        y_total = np.array(y_total)
      #  y_total = normalize_label(y_total)
        y_total = y_total[index]
        
        if task_id == 0:
            y_train = y_total[:train_num]   
            y_test = y_total[train_num:]
        else:
            y_train = y_total[:5]   #假如不是第一个阶段,那么训练数据应该是小样本
            y_test = y_total[5:10]
        #y_train = y_total[:train_num]
        return X_train,y_train,X_test,y_test
    
    def main(self):
        task_classes_dict = self.number_class_each_task
        Task = {i:[] for i in range(len(task_classes_dict))}
        X_test, y_test = None,None
        if type(self.ucr_dataset_base_folder)==list:
            for task_id,task in task_classes_dict.items():  
                X_train, y_train = None,None
                for flod_id in task:
                    X_train0, y_train0, X_test0, y_test0 = self.load_single_task_dataset(self.ucr_dataset_base_folder[flod_id],task_id)
                    X_train = X_train0 if X_train is None else np.concatenate((X_train0, X_train), axis=0)
                    y_train = y_train0 if y_train is None else np.concatenate((y_train0, y_train), axis=0)
                    X_test = X_test0 if X_test is None else np.concatenate((X_test0, X_test), axis=0)
                    y_test = y_test0 if y_test is None else np.concatenate((y_test0, y_test), axis=0)
                #print(X_train.shape,len(y_train),X_test.shape,len(y_test),'y_trainX_test***********')
                Task[task_id] = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}    
            return Task
        else:
            X_train, y_train, X_test, y_test = self.load_single_task_dataset(self.ucr_dataset_base_folder)
            return X_train, y_train
    
    
    