import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time, json
import os, sys

""" Training dataset"""

class DataSetIter(torch.utils.data.Dataset):
    def __init__(self, _base_img, _base_labels, _index2pos, _margin, _patch_size, _append_dim) -> None:
        self.base_img = _base_img #全量数据包括margin (145+2margin * 145+2margin * spe)
        self.base_labels = _base_labels #全量数据无margin (145 * 145)
        self.index2pos = _index2pos #训练数据 index -> (x, y) 对应margin后base_img的中心点坐标
        self.size = len(_index2pos)

        self.margin = _margin
        self.patch_size = _patch_size
        self.append_dim = _append_dim
    
    def __getitem__(self, index):
        start_x, start_y = self.index2pos[index]
        patch = self.base_img[start_x:start_x+2*self.margin+1 , start_y:start_y+2*self.margin+1,:]
        if self.append_dim:
            patch = np.expand_dims(patch, 0) # [channel=1, h, w, spe]
            patch = patch.transpose((0,3,1,2)) # [c, spe, h, w]
        else:
            patch = patch.transpose((2, 0, 1)) #[spe, h, w]
        label = self.base_labels[start_x, start_y] - 1
        # print(index, patch.shape, start_x, start_y, label)
        return torch.FloatTensor(patch), torch.LongTensor(label.reshape(-1))[0]

    def __len__(self):
        return self.size

    
class HSIDataLoader(object):
    def __init__(self, param) -> None:
        self.data_param = param['data']
        self.data_path_prefix = "../data"
        self.data = None #原始读入X数据 shape=(h,w,c)
        self.labels = None #原始读入Y数据 shape=(h,w,1)
        self.TR = None #标记训练数据
        self.TE = None #标记测试数据

        # 参数设置
        self.data_path_prefix = self.data_param.get('data_path_prefix', '../data')
        self.if_numpy = self.data_param.get('if_numpy', False)
        self.data_sign = self.data_param.get('data_sign', 'Indian')
        self.data_file = self.data_param.get('data_file', self.data_sign)
        self.patch_size = self.data_param.get('patch_size', 13) # n * n
        self.remove_zeros = self.data_param.get('remove_zeros', True)
        self.batch_size = self.data_param.get('batch_size', 256)
        self.none_zero_num = self.data_param.get('none_zero_num', 0)
        self.spectracl_size = self.data_param.get("spectral_size", 0)
        self.append_dim = self.data_param.get("append_dim", False)
        self.use_norm = self.data_param.get("use_norm", True)
        self.norm_type = self.data_param.get("norm_type", 'max_min') # 'none', 'max_min', 'mean_var'


        self.diffusion_data_path = self.data_param.get("diffusion_data_path", "unet3d_pavia.pkl")


    def load_data_from_diffusion(self, data_ori, labels):
        path = "%s/%s" % (self.data_path_prefix, self.diffusion_data_path)
        data = np.load(path)
        ori_h, ori_w, _= data_ori.shape
        h, w, _= data.shape
        assert ori_h == h, ori_w == w
        print("load diffusion data shape is ", data.shape)
        return data, labels 

    def load_raw_data(self):
        data, labels = None, None
        assert self.data_sign in ['Indian', 'Pavia', 'Houston', 'Salinas']
        data_path = '%s/%s/%s_split.mat' % (self.data_path_prefix, self.data_sign, self.data_file)
        all_data = sio.loadmat(data_path)
        data = all_data['input']
        TR = all_data['TR'] # train label
        TE = all_data['TE'] # test label
        labels = TR + TE
        return data, labels, TR, TE

    def load_data(self):
        ori_data, labels, TR, TE = self.load_raw_data()
        diffusion_data, diffusion_labels = self.load_data_from_diffusion(ori_data, labels)
        return diffusion_data, diffusion_labels, TR, TE

    def _padding(self, X, margin=2):
        # pading with zeros
        w,h,c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX
    
    def get_valid_num(self, y):
        tempy = y.reshape(-1)
        validy = tempy[tempy > 0]
        print('valid y shape is ', validy.shape)
        return validy.shape[0]

    def get_train_test_num(self, TR, TE):
        train_num, test_num = TR[TR>0].reshape(-1).size, TE[TE>0].reshape(-1).size
        print("train_num=%s, test_num=%s" % (train_num, test_num))
        return train_num, test_num

        
    def get_train_test_patches(self, X, y, TR, TE):
        h, w, c = X.shape
        # 给 X 做 padding
        windowSize = self.patch_size
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = self._padding(X, margin=margin)
        
        # 确定train和test的数据量
        train_num, test_num = self.get_train_test_num(TR, TE)
        trainX_index2pos = {}
        testX_index2pos = {}
        all_index2pos = {}

        patchIndex = 0
        trainIndex = 0
        testIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                start_x, start_y = r-margin, c-margin
                tempy = y[start_x, start_y]
                temp_tr = TR[start_x, start_y] 
                temp_te = TE[start_x, start_y]
                if temp_tr > 0 and temp_te > 0:
                    print("here", temp_tr, temp_te, r, c)
                    raise Exception("data error, find sample in trainset as well as testset.")

                if temp_tr > 0: #train data
                    trainX_index2pos[trainIndex] = [start_x, start_y]
                    trainIndex += 1
                elif temp_te > 0:
                    testX_index2pos[testIndex] = [start_x, start_y]
                    testIndex += 1
                all_index2pos[patchIndex] =[start_x, start_y]
                patchIndex = patchIndex + 1
        return zeroPaddedX, y, trainX_index2pos, testX_index2pos, all_index2pos, margin, self.patch_size 


    def applyPCA(self,   X, numComponents=30):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX

    def mean_var_norm(self, data):
        print("use mean_var norm...")
        h, w, c = data.shape
        data = data.reshape(h * w, c)
        data = StandardScaler().fit_transform(data)
        data = data.reshape(h, w, c)
        return data

    def data_preprocessing(self, data):
        '''
        1. normalization
        2. pca
        3. spectral filter
        data: [h, w, spectral]
        '''
        if self.norm_type == 'max_min':
            norm_data = np.zeros(data.shape)
            for i in range(data.shape[2]):
                input_max = np.max(data[:,:,i])
                input_min = np.min(data[:,:,i])
                norm_data[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
        elif self.norm_type == 'mean_var':
            norm_data = self.mean_var_norm(data)
        else:
            norm_data = data 
        pca_num = self.data_param.get('pca', 0)
        if pca_num > 0:
            print('before pca')
            pca_data = self.applyPCA(norm_data, int(self.data_param['pca']))
            norm_data = pca_data
            print('after pca')
        if self.spectracl_size > 0: # 按照给定的spectral size截取数据
            norm_data = norm_data[:,:,:self.spectracl_size]
        return norm_data



    def generate_numpy_dataset(self):
        #1. 根据data_sign load data
        self.data, self.labels, self.TR, self.TE = self.load_data()
        print('[load data done.] load data shape data=%s, label=%s' % (str(self.data.shape), str(self.labels.shape)))

        #2. 数据预处理 主要是norm化
        norm_data = self.data_preprocessing(self.data) 
        
        print('[data preprocessing done.] data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape))) 

        # 3. reshape & filter
        h, w, c = norm_data.shape
        norm_data = norm_data.reshape((h*w,c))
        norm_label = self.labels.reshape((h*w))
        TR_reshape = self.TR.reshape((h*w))
        TE_reshape = self.TE.reshape((h*w))
        TrainX = norm_data[TR_reshape>0]
        TrainY = norm_label[TR_reshape>0]
        TestX = norm_data[TE_reshape>0]
        TestY = norm_label[TE_reshape>0]
        train_test_data = norm_data[norm_label>0]
        train_test_label = norm_label[norm_label>0]
        
        print('------[data] split data to train, test------')
        print("X_train shape : %s" % str(TrainX.shape))
        print("Y_train shape : %s" % str(TrainY.shape))
        print("X_test shape : %s" % str(TestX.shape))
        print("Y_test shape : %s" % str(TestY.shape))

        return TrainX, TrainY, TestX, TestY, norm_data

    def reconstruct_pred(self, y_pred):
        '''
        根据原始label信息 对一维预测结果重建图像
        y_pred: [h*w]
        return: pred: [h, w]
        '''
        h, w = self.labels.shape
        return y_pred.reshape((h,w))

    def prepare_data(self):
        #1. 根据data_sign load data
        self.data, self.labels, self.TR, self.TE = self.load_data()
        print('[load data done.] load data shape data=%s, label=%s' % (str(self.data.shape), str(self.labels.shape)))

        #2. 数据预处理 主要是norm化
        norm_data = self.data_preprocessing(self.data) 
        
        print('[data preprocessing done.] data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))

        #3. 获取patch 并形成batch型数据
        base_img, labels, train_index2pos, test_index2pos, all_index2pos, margin, patch_size \
              = self.get_train_test_patches(norm_data, self.labels, self.TR, self.TE)

        print('------[data] split data to train, test------')
        print("train len: %s" % len(train_index2pos ))
        print("test len : %s" % len(test_index2pos ))
        print("all len: %s" % len(all_index2pos ))


        trainset = DataSetIter(base_img, labels, train_index2pos, margin, patch_size, self.append_dim) 
        unlabelset=DataSetIter(base_img,labels,test_index2pos,margin, patch_size, self.append_dim)
        testset = DataSetIter(base_img, labels, test_index2pos , margin, patch_size, self.append_dim) 
        allset = DataSetIter(base_img, labels, all_index2pos, margin, patch_size, self.append_dim) 
        
        return trainset, unlabelset, testset, allset
 
    def generate_torch_dataset(self):
        # 0. 判断是否使用numpy数据集
        if self.if_numpy:
            return self.generate_numpy_dataset()

        
        trainset, unlabelset, testset, allset = self.prepare_data()

        multi=self.data_param.get('unlabelled_multiple',1)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                drop_last=False
                                                )
        unlabel_loader=torch.utils.data.DataLoader(dataset=unlabelset,
                                                batch_size=int(self.batch_size*multi),
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                                )
        all_loader = torch.utils.data.DataLoader(dataset=allset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                                )
        
        return train_loader, unlabel_loader,test_loader, all_loader

       


if __name__ == "__main__":
    dataloader = HSIDataLoader({"data":{"data_path_prefix":'../../data', "data_sign": "Indian",
        "data_file": "Indian_40"}})
    train_loader, unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset()
