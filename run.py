# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 19:45:49 2022
@author: Lenovo
"""
from main import main
import gc

'''
数据集有3种类型：
    1、铝电解数据集：槽电压数据，不同槽作为不同的task
       ucr_dataset_name = 'Alu'
       ucr_dataset_base_folder = [path_volt_8310_3classes,
                                path_volt_8311_3classes,
                                path_volt_8312_3classes,
                                path_volt_8331_3classes], 
   
    2、TEP数据集
       ucr_dataset_name = 'TEP'
       ucr_dataset_base_folder = [
                                 ucr_path_mode1,
                                 ucr_path_mode3
                                 ]
'''
def iteration(k_list,c_list,lmin_list):
    if type(k_list)==list and type(c_list)==float:
        for k in k_list:
            main0 = main(
                        ucr_dataset_name = 'TEP', # 要么是MedicalImages,要么是Alu,TEP
                       # ucr_dataset_name = 'MedicalImages', # 要么是MedicalImages,要么是Alu,TEP
                        #ucr_dataset_name = ['NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2'],
            #            ucr_dataset_base_folder = [
            #                                       context1,
            #                                       context2,
            #                                      # context3,
            #                                       #context4,
            #                                       context5,
            #                                       context6,
            #                                       #context7,
            #                                       context8,
            #                                       ], 
                       pre_train_dataset_folder = path_pre_training,  
            #            ucr_dataset_base_folder = [
            #                                       ucr_path_mode1,
            #                                       ucr_path_mode3
            #                                       ],
            #            ucr_dataset_base_folder = [
            #                                       ucr_path_task1,
            #                                       #ucr_path_task2,
            #                                       ucr_path_task3,
            #                                       ucr_path_task4,
            #                                       ucr_path_task5,#
            #                                       
            #                                       #ucr_path_task6,
            #                                       ucr_path_task7,
            #                                       ucr_path_task8,
            #                                       ucr_path_task9,
            #                                       ucr_path_task10
            #                                       ],
                     #   ucr_dataset_base_folder = path_UCR_2018,
                        ucr_dataset_base_folder = [task0_mode3,
                                                    task1_mode3,
                                                    task2_mode3,
                                                    task3_mode3,
                                                    task4_mode3,
                                                    task5_mode3,
                                                    task6_mode3,
                                                    task7_mode3,
                                                    task8_mode3,
                                                    task9_mode3,
                                                    task10_mode3,
                                                    task11_mode3,
                                                    task12_mode3,
                                                    task13_mode3,
                                                    task14_mode3,
                                                    task15_mode3],  
#                        ucr_dataset_base_folder = [
#                                               task0,
#                                               task1,
#                                               task2,
#                                               task3,
#                                               task4,
#                                               task5,
#                                              ],
                        K = k, 
                        Lmin = 0.2, 
                        temp_factor = 0.08,
                        pre_train_temp_factor = 0.08,
                        learning_rate = 0.001,
                        learning_rate_meta = 0.001,
                        epoch = 10, 
                        epoch_meta_learning = 100,
                        lw = 0.00005,
                        is_incremental_learning = True,
                       # number_class_each_task = {0:[1,2,3,4,5,6],1:[7,8],2:[9,10]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                        number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7],2:[8,9],3:[10,11],4:[12,13],5:[14,15]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                        #number_class_each_task = {0:[0,1,2,3,4,5],1:[6],2:[7],3:[8],4:[9],5:[10],6:[11],7:[12],8:[13],9:[14],10:[15]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                       # number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7,8,9],2:[10,11,12,13]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]} 
                       # number_class_each_task = {0:[0,1,2,3],1:[4],2:[5]},
                        variable_selection = 21,  #3/16 代表 XMEAS4/XMEAS17
                        add_pre_trained = False,
                        dropout = 0.2,
                        c = 0.5
                        )
            print(gc.get_stats(),'before___clear')
            main0.incremental_learning_train()
            gc.collect()
            print(gc.get_stats())
    elif type(c_list)==list and type(k_list)==float:
        for c in c_list:
            main0 = main(
                        ucr_dataset_name = 'Alu', # 要么是MedicalImages,要么是Alu,TEP
                       # ucr_dataset_name = 'MedicalImages', # 要么是MedicalImages,要么是Alu,TEP
                        #ucr_dataset_name = ['NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2'],
            #            ucr_dataset_base_folder = [
            #                                       context1,
            #                                       context2,
            #                                      # context3,
            #                                       #context4,
            #                                       context5,
            #                                       context6,
            #                                       #context7,
            #                                       context8,
            #                                       ], 
                       pre_train_dataset_folder = path_pre_training,  
            #            ucr_dataset_base_folder = [
            #                                       ucr_path_mode1,
            #                                       ucr_path_mode3
            #                                       ],
            #            ucr_dataset_base_folder = [
            #                                       ucr_path_task1,
            #                                       #ucr_path_task2,
            #                                       ucr_path_task3,
            #                                       ucr_path_task4,
            #                                       ucr_path_task5,#
            #                                       
            #                                       #ucr_path_task6,
            #                                       ucr_path_task7,
            #                                       ucr_path_task8,
            #                                       ucr_path_task9,
            #                                       ucr_path_task10
            #                                       ],
                     #   ucr_dataset_base_folder = path_UCR_2018,
#                        ucr_dataset_base_folder = [task0_mode3,
#                                                    task1_mode3,
#                                                    task2_mode3,
#                                                    task3_mode3,
#                                                    task4_mode3,
#                                                    task5_mode3,
#                                                    task6_mode3,
#                                                    task7_mode3,
#                                                    task8_mode3,
#                                                    task9_mode3,
#                                                    task10_mode3,
#                                                    task11_mode3,
#                                                    task12_mode3,
#                                                    task13_mode3,
#                                                    task14_mode3,
#                                                    task15_mode3],  
                        ucr_dataset_base_folder = [
                                               task0,
                                               task1,
                                               task2,
                                               task3,
                                               task4,
                                               task5,
                                              ],
                        K = 0.3, 
                        Lmin = 0.3, 
                        temp_factor = 0.08,
                        pre_train_temp_factor = 0.08,
                        learning_rate = 0.001,
                        learning_rate_meta = 0.001,
                        epoch = 100, 
                        epoch_meta_learning = 200,
                        lw = 0.00005,
                        is_incremental_learning = True,
                        #number_class_each_task = {0:[1,2,3,4,5,6],1:[7,8],2:[9,10]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                        #number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7],2:[8,9],3:[10,11],4:[12,13],5:[14,15]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                        #number_class_each_task = {0:[0,1,2,3,4,5],1:[6],2:[7],3:[8],4:[9],5:[10],6:[11],7:[12],8:[13],9:[14],10:[15]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                       # number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7,8,9],2:[10,11,12,13]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]} 
                        number_class_each_task = {0:[0,1,2,3],1:[4],2:[5]},
                        variable_selection = 21,  #3/16 代表 XMEAS4/XMEAS17
                        add_pre_trained = False,
                        dropout = 0.2,
                        c = c
                        )
            main0.incremental_learning_train()
    elif type(lmin_list)==list and type(k_list)==float and type(c_list)==float:
        for lmin in lmin_list:
            main0 = main(
                        ucr_dataset_name = 'Alu', # 要么是MedicalImages,要么是Alu,TEP
                       # ucr_dataset_name = 'MedicalImages', # 要么是MedicalImages,要么是Alu,TEP
                        #ucr_dataset_name = ['NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2'],
            #            ucr_dataset_base_folder = [
            #                                       context1,
            #                                       context2,
            #                                      # context3,
            #                                       #context4,
            #                                       context5,
            #                                       context6,
            #                                       #context7,
            #                                       context8,
            #                                       ], 
                       pre_train_dataset_folder = path_pre_training,  
            #            ucr_dataset_base_folder = [
            #                                       ucr_path_mode1,
            #                                       ucr_path_mode3
            #                                       ],
            #            ucr_dataset_base_folder = [
            #                                       ucr_path_task1,
            #                                       #ucr_path_task2,
            #                                       ucr_path_task3,
            #                                       ucr_path_task4,
            #                                       ucr_path_task5,#
            #                                       
            #                                       #ucr_path_task6,
            #                                       ucr_path_task7,
            #                                       ucr_path_task8,
            #                                       ucr_path_task9,
            #                                       ucr_path_task10
            #                                       ],
                     #   ucr_dataset_base_folder = path_UCR_2018,
#                        ucr_dataset_base_folder = [task0_mode3,
#                                                    task1_mode3,
#                                                    task2_mode3,
#                                                    task3_mode3,
#                                                    task4_mode3,
#                                                    task5_mode3,
#                                                    task6_mode3,
#                                                    task7_mode3,
#                                                    task8_mode3,
#                                                    task9_mode3,
#                                                    task10_mode3,
#                                                    task11_mode3,
#                                                    task12_mode3,
#                                                    task13_mode3,
#                                                    task14_mode3,
#                                                    task15_mode3],  
                        ucr_dataset_base_folder = [
                                               task0,
                                               task1,
                                               task2,
                                               task3,
                                               task4,
                                               task5,
                                              ],
                        K = 0.3, 
                        Lmin = lmin, 
                        temp_factor = 0.08,
                        pre_train_temp_factor = 0.08,
                        learning_rate = 0.001,
                        learning_rate_meta = 0.001,
                        epoch = 100, 
                        epoch_meta_learning = 200,
                        lw = 0.00005,
                        is_incremental_learning = True,
                       # number_class_each_task = {0:[1,2,3,4,5,6],1:[7,8],2:[9,10]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                        #number_class_each_task = {0:[0,1],1:[2],2:[3]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                       # number_class_each_task = {0:[0,1,2,3,4,5],1:[6],2:[7],3:[8],4:[9],5:[10],6:[11],7:[12],8:[13],9:[14],10:[15]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                        #number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7,8,9],2:[10,11,12,13]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]} 
                       # number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7,8],2:[9,10,11],3:[12,13,14]},
                        number_class_each_task = {0:[0,1,2,3],1:[4],2:[5]},
                        variable_selection = 21,  #3/16 代表 XMEAS4/XMEAS17
                        add_pre_trained = False,
                        dropout = 0.2,
                        c = 0.5
                        )
            
            main0.incremental_learning_train()
    else:
        main0 = main(
                    ucr_dataset_name = 'TEP', # 要么是MedicalImages,要么是Alu,TEP
                   # ucr_dataset_name = 'MedicalImages', # 要么是MedicalImages,要么是Alu,TEP
                    #ucr_dataset_name = ['NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2'],
        #            ucr_dataset_base_folder = [
        #                                       context1,
        #                                       context2,
        #                                      # context3,
        #                                       #context4,
        #                                       context5,
        #                                       context6,
        #                                       #context7,
        #                                       context8,
        #                                       ], 
                   pre_train_dataset_folder = path_pre_training,  
        #            ucr_dataset_base_folder = [
        #                                       ucr_path_mode1,
        #                                       ucr_path_mode3
        #                                       ],
        #            ucr_dataset_base_folder = [
        #                                       ucr_path_task1,
        #                                       #ucr_path_task2,
        #                                       ucr_path_task3,
        #                                       ucr_path_task4,
        #                                       ucr_path_task5,#
        #                                       
        #                                       #ucr_path_task6,
        #                                       ucr_path_task7,
        #                                       ucr_path_task8,
        #                                       ucr_path_task9,
        #                                       ucr_path_task10
        #                                       ],
                 #   ucr_dataset_base_folder = path_UCR_2018,
                    ucr_dataset_base_folder = [task0_mode3,
                                                task1_mode3,
                                                task2_mode3,
                                                task3_mode3,
                                                task4_mode3,
                                                task5_mode3,
                                                task6_mode3,
                                                task7_mode3,
                                                task8_mode3,
                                                task9_mode3,
                                                task10_mode3,
                                                task11_mode3,
                                                task12_mode3,
                                                task13_mode3,
                                                task14_mode3,
                                                task15_mode3],  
#                    ucr_dataset_base_folder = [
#                                               task0,
#                                               task1,
#                                               task2,
#                                               task3,
#                                               task4,
#                                               task5,
#                                              ],
                    K = 0.3, 
                    Lmin = 0.2, 
                    temp_factor = 0.08,
                    pre_train_temp_factor = 0.08,
                    learning_rate = 0.001,
                    learning_rate_meta = 0.001,
                    epoch = 10, 
                    epoch_meta_learning = 100,
                    lw = 0.00005,
                    is_incremental_learning = True,
                   # number_class_each_task = {0:[1,2,3,4,5,6],1:[7,8],2:[9,10]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                    number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7],2:[8,9],3:[10,11],4:[12,13],5:[14,15]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                   # number_class_each_task = {0:[0,1,2,3,4,5],1:[6],2:[7],3:[8],4:[9],5:[10],6:[11],7:[12],8:[13],9:[14],10:[15]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]}
                    #number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7,8,9],2:[10,11,12,13]},   #每个任务有几个类别/用自己用字典定义一个类别数分配规律。{1:[1,2,3,4,5,6],2:[7,8],3:[9,10]} 
                   # number_class_each_task = {0:[0,1,2,3,4,5],1:[6,7,8],2:[9,10,11],3:[12,13,14]},
                   # number_class_each_task = {0:[0,1,2,3],1:[4],2:[5]},
                    variable_selection = 21,  #3/16 代表 XMEAS4/XMEAS17
                    add_pre_trained = False,
                    dropout = 0.2,
                    c = 0.5
                    )
        
        main0.incremental_learning_train()
    return

#k,
iteration(0.3,0.7,0.2)