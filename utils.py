# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:49:15 2022

@author: Lenovo
"""

from torch import nn
from os import path
from matplotlib import pyplot
from matplotlib import cm
from numpy import genfromtxt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans

import torch
import random
import numpy as np
import seaborn as sns
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

'''--------------------------------------------------Eval Accuracy-------------------------------------------'''

def eval_accuracy(model, X, Y, transformer, Prototypes):
    predictions, Prototypes, result_before, Prototypes_before  = model.predict(X, transformer, Prototypes)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1) 
    print(predictions,Y,result_before,'**************')
    
    Accuracy = (predictions == Y).sum() / Y.size
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    
#    Accuracy = (predictions_view1 == Y).sum() / Y.size
#    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
#    
#    Accuracy = (predictions_view2 == Y).sum() / Y.size
#    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
#    
#    Accuracy = (predictions_view3 == Y).sum() / Y.size
#    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    
    Accuracy_before = (result_before == Y).sum() / Y.size
    print(f"Accuracy_before_transformer: {(result_before == Y).sum() / Y.size}")
    return Accuracy, Prototypes


'''--------------------------------------------------Data Normalize-------------------------------------------'''

def normalize_data_0_1(X):
    #shape = X.shape
    normalizeData = (X-np.min(X))/(np.max(X)-np.min(X))
    return normalizeData

def Znormalize_tensor(X):
    X_mean=X.mean()
    X_std=X.std()
    return torch.div(torch.sub(X,X_mean),X_std)

def Znormalize_dim2(X,dim):
    #dim:表示在哪个维度上进行归一化
    #the dim of X is 3
    Y=torch.zeros(X.shape[0],X.shape[1])
    if dim==1:
        for i in range(X.shape[1]):
            Y[:,i]=Znormalize_tensor(X[:,i])
    if dim==0:
        for i in range(X.shape[0]):
            Y[i,:]=Znormalize_tensor(X[i,:])
    return Y

def normalize_label(label):
    '''
       将标签变为从0开始的自然数
    '''
    label_tensor = torch.tensor(label)
    label_tensor = label_tensor.float()
    for i in list(set(label)):
        label_tensor = torch.where(label_tensor==i,torch.Tensor([list(set(label)).index(i)]).expand(label_tensor.shape[0]),label_tensor)
    label = label_tensor.numpy()
    return label

def load_UCR(ucr_dataset_path,ucr_dataset_name):
    '''对2015版本的UCR数据集和2018版本的数据集分别加载'''
    if '2015' in ucr_dataset_path:
        dataset_path = path.join(ucr_dataset_path, ucr_dataset_name)
        train_file_path = path.join(dataset_path, '{}_TRAIN'.format(ucr_dataset_name))
        test_file_path = path.join(dataset_path, '{}_TEST'.format(ucr_dataset_name))
        train_raw_arr = genfromtxt(train_file_path, delimiter=',')
        test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    else:
        dataset_path = path.join(ucr_dataset_path, ucr_dataset_name)
        train_file_path = path.join(dataset_path, '{}_TRAIN.tsv'.format(ucr_dataset_name))
        test_file_path = path.join(dataset_path, '{}_TEST.tsv'.format(ucr_dataset_name))
        train_raw_arr = np.loadtxt(train_file_path, delimiter="\t")
        test_raw_arr = np.loadtxt(test_file_path, delimiter='\t')
    return train_raw_arr,test_raw_arr

def normalize_y(y_train,y_test):
     #当标签不是0,1,2这种，而是1,2这种时
    if min(y_train)!=0:
        y_train = y_train-min(y_train)
        y_test = y_test-min(y_test)
    #当数据标签是0,2这种时
    if max(y_train)-min(y_train) >= len(set(y_train)):
        asign = lambda t: t-(max(y_train)-len(set(y_train))+1) if t != min(y_train) else min(y_train)
        y_train = list(map(asign, y_train))
        y_test = list(map(asign, y_test))
        y_train = np.array(y_train)
        y_test = np.array(y_test)
    return y_train,y_test

def normalize_standard(X, scaler=None):
    shape = X.shape
    data_flat = X.flatten() #把数据降到1维
    if scaler is None:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(np.product(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(np.product(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, scaler=None):
    if type(X)==torch.Tensor:
        X = X.numpy()
    else:
        X = X
    if scaler is None:
        X, scaler = normalize_standard(X)
    else:
        X, scaler = normalize_standard(X, scaler)
    return X, scaler

def preprocessing_standard0(L):
    """
       preprocessing_standard: 对一行数据进行数据标准化,L[i]=(L[i]-L.min)/(L.max-L.min)
    """
    datamax=max(L)
    datamin=min(L)
    L1=[]
    for index,row in enumerate(L):
        if datamax-datamin!=0:
            m=(row-datamin)/float((datamax-datamin))
            L1.append(round(m,4))
        else:
            L1.append(0)
   #        matlabshow(row,index=str(index)+'_')    
    return L1
    
def preprocess0(dataframe):
    dataframe_new = pd.DataFrame(columns=dataframe.columns)
    for i in dataframe.columns:
        dataframe_new[i] = preprocessing_standard0(dataframe[i])
        #dataframe_new[i]=pywt_pro_0(list(dataframe_new[i]))
    return dataframe_new

'''--------------------------------------------------kmeans for initial shapelet-------------------------------------------'''

def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    
    '''因为n_channels为24，但是只想要一个通道的shapelet，所以改为了1'''
    segments = np.empty((n_segments, 1, shapelets_size))
    for i, k in enumerate(samples_i):
        samples_dim = random.choices(range(n_channels), k=1)
        s = random.randint(0, len_ts - shapelets_size)\
        #s=15
        segments[i] = X[k, samples_dim, s:s+shapelets_size]
    return segments

def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    #print(segments.shape,'segments.shape')
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    #print(clusters.shape,'clusters.shape')
    return clusters
    
'''--------------------------------------------------plot shapelet-------------------------------------------'''
    
def torch_dist_ts_shapelet(ts, shapelet, cuda=False):
    """
    Calculate euclidean distance of shapelet to a time series via PyTorch and returns the distance along with the position in the time series.
    """
    if not isinstance(ts, torch.Tensor):
        ts = torch.tensor(ts, dtype=torch.float)
    if not isinstance(shapelet, torch.Tensor):
        shapelet = torch.tensor(shapelet, dtype=torch.float)
    if cuda:
        ts = ts.cuda()
        shapelet = shapelet.cuda()
    shapelet = shapelet[:1,:]
    shapelet = torch.unsqueeze(shapelet, 1)
    ts = ts.unfold(1,shapelet.shape[2], 1)
    dists = torch.cdist(ts, shapelet)
    if dists.shape[0]>1:
        #阳极电流数据是多维的，min_single_dim是子序列与单个序列的匹配位置，min_total_dim是子序列最匹配的维度
        min_num,min_single_dim = torch.min(dists, dim=1)
        d_min, min_total_dim = torch.min(min_num, 0)
        return (min_single_dim[min_total_dim.item()].item(), min_total_dim.item())
    else:
        #公共数据
        dists = torch.sum(dists, dim=0)
        d_min,d_argmin = torch.min(dists, dim=0)
        return (d_min.item(), d_argmin.item())

def lead_pad_shapelet(shapelet, pos):
    """
    Adding leading NaN values to shapelet to plot it on a time series at the best matching position.
    """
    pad = np.empty(pos)
    pad[:] = np.NaN
    padded_shapelet = np.concatenate([pad, shapelet])
    return padded_shapelet

def record_shapelet_value(view,ucr_dataset_name,shapelets,X_test,pos,i,j):
    '''将shapelet的值保存到文档里面'''
    path='results/shapelet_value/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    path_shapelet_value=path='results/shapelet_value/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'+str(view)+'/'+'shapelet'+str(j)+'/'
    if not os.path.exists(path_shapelet_value):
        os.mkdir(path_shapelet_value)
    excel_shapelet=lead_pad_shapelet(shapelets[j, 0], pos)
    if X_test.shape[1]>1:
        excel_time_series=X_test[i,pos]
    else:
        excel_time_series=X_test[i]
    excel_shapelet=pd.DataFrame(excel_shapelet)
    excel_time_series=pd.DataFrame(excel_time_series)
    excel_shapelet.to_excel(path_shapelet_value+'shapelet.xlsx')
    excel_time_series.to_excel(path_shapelet_value+'time_series.xlsx')
    return

def plot_sub(view,i,j,fig,shapelets,X_test,record_data_plot,test_y,ucr_dataset_name):
    '''
    i:num of sample
    j:num of sub_graph
    shapelets:[num of shapelets,1, len_of_shapelets]
    X_test:[num of test,1, len of test]
    '''
    font = {'family': 'Times New Roman',
        'style': 'normal',
        'stretch': 1000,
        'weight': 'bold',
        }
    fig_ax1 = fig.add_subplot(4,int(shapelets.shape[0]/4)+1,j+1)
    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=None, wspace=0.3, hspace=0.5)#wspace 子图横向间距， hspace 代表子图间的纵向距离，left 代表位于图像不同位置
    fig_ax1.text(0.01,0.01,'',fontdict=font)
    fig_ax1.set_title("shapelet"+str(j+1),fontproperties="Times New Roman",)
    if X_test.shape[1]>1:
        _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[j])
        fig_ax1.plot(X_test[i, pos], color='black', alpha=0.02, )
        fig_ax1.plot(lead_pad_shapelet(shapelets[j, 0], _), color='#F03613', alpha=0.02)
        record_shapelet_value(view,ucr_dataset_name,shapelets,X_test,pos,i,j)
    else:
        fig_ax1.plot(X_test[i, 0], color='black', alpha=0.5)
        _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[j])
        fig_ax1.plot(lead_pad_shapelet(shapelets[j, 0], pos), color='#F03613', alpha=0.5)
        record_shapelet_value(view,ucr_dataset_name,shapelets,X_test,pos,i,j)
    record_data_plot['fig'+str(j)]['x']=X_test[i, 0]
    record_data_plot['fig'+str(j)]['s']=shapelets[j, 0]
    record_data_plot['fig'+str(j)]['class']=test_y[i]
    record_data_plot['fig'+str(j)]['dim']=pos
    return record_data_plot

def featrue_map(shapelet_transform, y_test, weights, ucr_dataset_name, X_test, shapelet_num):
    '''设置全局字体'''
    pyplot.rcParams['font.sans-serif']='Times New Roman'
    pyplot.rcParams['font.weight']='bold'
    pyplot.rcParams['font.size']=14
    pyplot.rc('xtick',labelsize=10)
    pyplot.rc('ytick',labelsize=10)
    
    fig = pyplot.figure(facecolor='white')
    #fig.set_size_inches(20, 8)
    gs = gridspec.GridSpec(2, 2)
    fig_ax3 = fig.add_subplot(gs[:, :])
    #font0 = FontProperties(family='serif',weight='bold',size=14)
    #fig_ax3.set_title("The decision boundaries learned by the model to separate the two classes.", fontproperties=font0)
    color = {-1:'#00FF00',0: '#F03613', 1: '#7BD4CC', 2: '#00281F', 3: '#BEA42E',4:'#FFC0CB',5:'#FFF0F5',6:'#FF69B4'}
             
    dist_s1=shapelet_transform[:,shapelet_num]
    dist_s2=shapelet_transform[:,shapelet_num+1]
    fig_ax3.scatter(dist_s1, dist_s2, color=[color[l] for l in y_test])
    
    # Create a meshgrid of the decision boundaries
    xmin = np.min(shapelet_transform[:, shapelet_num]) - 0.1
    xmax = np.max(shapelet_transform[:, shapelet_num]) + 0.1
    ymin = np.min(shapelet_transform[:, shapelet_num+1]) - 0.1
    ymax = np.max(shapelet_transform[:, shapelet_num+1]) + 0.1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin)/200),
                            np.arange(ymin, ymax, (ymax - ymin)/200))
    Z = []
    num_class=len(weights)
    for x, y in np.c_[xx.ravel(), yy.ravel()]:
        Z.append(np.argmax([weights[i][0]*x + weights[i][1]*y
                               for i in range(num_class)]))
   # Z = numpy.array(Z).reshape(xx.shape)
    #fig_ax3.contourf(xx, yy, Z / 3, cmap=viridis, alpha=0.25)
    fig_ax3.set_xlabel("shapelet"+str(shapelet_num))
    fig_ax3.set_ylabel("shapelet"+str(shapelet_num+1))
    fig_ax3.tick_params(labelsize=13)
    
    path='shapelets_plots/'+str(ucr_dataset_name)+'shapelet'+str(shapelet_num)+'_'+str(shapelet_num+1)+'feature_map'+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    pyplot.savefig(path+'.pdf',format='pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    
    #pyplot.savefig(path+'.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
    return 

def plot_shapelets(X_test, shapelets, y_test, shapelet_transform, ucr_dataset_name, shapelets_size_and_len):
   #print(shapelets.shape,'&&&&&&&&&&shapelets.shape')
    num_shapelets = shapelets.shape[0]
   
    dist0 = {}  #第0个视图的数据
    dist1 = {}  #第1个视图的数据
    dist2 = {}  #第2个视图的数据
   # print(shapelet_transform.shape,'shapelet_transform.shape**********')
    nums_shapelets = shapelet_transform.shape[2]
    for i in range(nums_shapelets):
        dist0[i] = shapelet_transform[:, 0, i]
    for i in range(nums_shapelets):
        dist1[i] = shapelet_transform[:, 1, i]
    for i in range(nums_shapelets):
        dist2[i] = shapelet_transform[:, 2, i]
   # gs = gridspec.GridSpec(12, 8)
    #fig_ax1 = fig.add_subplot(gs[0:3, :4])
    record_data_plot = {}
    for i in range(nums_shapelets):
        record_data_plot['fig'+str(i)]={}
   # fig_ax1.set_title("top of its 1 best matching time series.") 
    '''第一个视图'''
    fig = pyplot.figure(facecolor='white')
    fig.set_size_inches(20, 8)
    for j in range(int(num_shapelets/3)):
        for i in np.argsort(dist0[j])[:1]:
            record_data_plot = plot_sub(1,i,j,fig,shapelets[:int(num_shapelets/3),:,:list(shapelets_size_and_len.keys())[0]],X_test,record_data_plot,y_test,ucr_dataset_name)
    
    '''第二个视图'''
    fig = pyplot.figure(facecolor='white')
    fig.set_size_inches(20, 8)
    for j in range(int(num_shapelets/3)):
        for i in np.argsort(dist1[j])[:1]:
            record_data_plot = plot_sub(2,i,j,fig,shapelets[int(num_shapelets/3):int(num_shapelets/3)*2,:,:list(shapelets_size_and_len.keys())[1]],X_test,record_data_plot,y_test,ucr_dataset_name)
    
    '''第三个视图'''
    fig = pyplot.figure(facecolor='white')
    fig.set_size_inches(20, 8)
    for j in range(int(num_shapelets/3)):
        for i in np.argsort(dist2[j])[:1]:
            record_data_plot = plot_sub(3,i,j,fig,shapelets[int(num_shapelets/3)*2:,:,:list(shapelets_size_and_len.keys())[2]],X_test,record_data_plot,y_test,ucr_dataset_name)
#    
    caption = """Shapelets learned for the pot volt dataset plotted on top of the best matching time series."""
    pyplot.figtext(0.5, -0.02, caption, wrap=True, horizontalalignment='center', fontsize=20, family='Times New Roman')
    path='results/shapelets_plots/'+str(ucr_dataset_name)+str(X_test.shape[0])+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    pyplot.savefig(path+'.pdf', format='pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    #pyplot.savefig(path+'.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
    pyplot.show()
    return record_data_plot

'''--------------------------------------------------Visualize Data-------------------------------------------'''
def visualization2D(transformed_x, text):
    '''
    x : numpy(number,dim)
    y : numpy(number)
    '''
    #画图

    fig = plt.figure(figsize=(4,4))
    font0 = {'family':'serif','weight':'bold','size':'20'}#定义图的字体
    plt.xlabel("dimention0", font0)
    plt.ylabel("dimention1", font0)
    plt.tick_params(labelsize=20)
    sns.set(font_scale=1.3)
    sns.scatterplot(data = transformed_x, hue='label', x='dim0', y='dim1')
    
    #保存图
    path='results/shapelets_plots/'+'visualization'+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path+text+'times'+'.pdf', facecolor=fig.get_facecolor(), bbox_inches="tight")
    transformed_x.to_excel(path+text+'times.xlsx')
    return

def visualization(x,y):
    #数据格式更改
    #训练x
    tsne = TSNE(n_components=2)
    tsne.fit(x)
    #转换x
    transformed_x = tsne.fit_transform(x)
    transformed_x = pd.DataFrame(transformed_x)
    transformed_x = transformed_x.rename(columns={0:'dim0',1:'dim1'})
    transformed_x['label'] = y 
    return transformed_x

def visualize_2D(x, y_predict, title):
    '''
     可视化数据
     
     参数：
     --------------------
     x；输入样本 tensor 
       [num_sample,dim]
    
     y_predict: 标签 tensor
        [num_sample]
    '''
    transformed_x_y = visualization(x.numpy(), y_predict.detach().numpy())
    visualization2D(transformed_x_y, title)
    return